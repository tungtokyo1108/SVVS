#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:17:59 2021

@author: tungdang
"""

import warnings
from abc import ABCMeta, abstractmethod
from time import time 

import math
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma, gammaln, logsumexp
from scipy import linalg
from joblib import Parallel, delayed

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn import cluster
from sklearn.utils.extmath import row_norms


def sigmoid(x):
    "Numerically stable sigmoid function."
    n_samples, n_features = x.shape
    z = np.empty((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            if x[i][j] >= 0:
                z[i][j] = np.exp(-x[i][j])
                z[i][j] = 1 / (1 + z[i][j])
            else:
                z[i][j] = np.exp(x[i][j])
                z[i][j] = z[i][j] / (1 + z[i][j])
    return z

"""
def sigmoid(x,z,i,j):
    "Numerically stable sigmoid function."
    
    if x[i][j] >= 0:
        z[i][j] = np.exp(-x[i][j])
        z[i][j] = 1 / (1 + z[i][j])
    else:
        z[i][j] = np.exp(x[i][j])
        z[i][j] = z[i][j] / (1 + z[i][j])
        
    return z[i][j]

def sigmoid_parallel(x):
    
    n_samples, n_features = x.shape
    z = np.empty((n_samples, n_features))
    parallel = Parallel(n_jobs = -1, verbose = 1, pre_dispatch = '2*n_jobs', prefer="processes")
    out = parallel(delayed(sigmoid)(x, z, i, j)for i, j in zip(range(n_samples),range(n_features)))
        
    return out

output = sigmoid_parallel(X)
"""

#------------------------------------------------------------------------------
# Stochastic Variational Inferenece for Dirichlet-multinomial mixture models 
#------------------------------------------------------------------------------

class DMM_VFS():
    
    def __init__(self, n_components=1, tol=1e-3, max_iter=100, n_init=1,
                 init_params='kmeans', weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None, weights_init=None, 
                 alpha=None, beta=None, 
                 random_state=42, warm_start=False, verbose=0, verbose_interval=10):
        
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start 
        self.verbose = verbose 
        self.verbose_interval = verbose_interval
        
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior 
        self.weights_init = weights_init 
        self.alpha = alpha 
        self.beta = beta 
        
    def _initialize_parameters(self, X, random_state):
        
        n_samples, n_features = X.shape
        
        self.weight_concentration_prior = 1./self.n_components
        self.select_prior = 1 
        
        self.gamma = np.ones((self.n_components, n_features))
        self.delta = np.ones((self.n_components, n_features))
        self.iota = np.ones(n_features)
        self.kappa = np.ones(n_features)
        
        random_state = check_random_state(self.random_state)
        
        self.resp = np.zeros((n_samples, self.n_components))
        
        if self.init_params == "kmeans":
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            self.resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            self.resp = random_state.rand(n_samples, self.n_components)
            self.resp /= self.resp.sum(axis=1)[:, np.newaxis]
            
        select = np.zeros((n_samples, n_features))
        reject = np.zeros((n_samples, n_features))
        for d in range(n_features):
            chois = random_state.rand(n_samples, 2)
            select[:,d] = chois[:,0]
            reject[:,d] = chois[:,1]
        select_norm = select/(select + reject)
        self.selected = select_norm
        
        nk = np.dot(self.resp.T, self.selected) + 10 * np.finfo(self.resp.dtype).eps
        self.gamma_vi = self.gamma 
        self.delta_vi = self.delta 
        self.iota_vi = self.iota 
        self.kappa_vi = self.kappa 
        
        self._estimate_weights()
        self._estimate_selections()
        
        return self.resp, self.selected
        
    def _estimate_weights(self):
        
        nk = self.resp.sum(axis=0) + 10 * np.finfo(self.resp.dtype).eps
        
        self.weight_concentration_ = (
            1. + nk,
            (self.weight_concentration_prior + 
             np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
    
    def _estimate_selections(self):
        
        self.xi1 = self.select_prior + self.selected.sum(axis=0)
        self.xi2 = self.select_prior + (1 - self.selected).sum(axis=0)
        
    def _estimate_alpha(self, X):
        
        n_samples, n_features = X.shape
        means_ = self.gamma_vi / self.gamma_vi.sum(axis=1)[:,np.newaxis]

        part_1 = np.dot(self.resp.T, self.selected) * means_ * digamma(means_.sum(axis=1))[:, np.newaxis]
        
        dig_sum_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            dig_sum_1[:,k] = np.sum(y, axis=1)
        part_2 = np.dot((self.resp * digamma(dig_sum_1)).T, self.selected) * means_
        
        part_3 = np.empty((self.n_components, n_features))
        for k in range(self.n_components):
            y = X + means_[k]
            y = self.selected * digamma(y) * self.resp[:,k][:, np.newaxis]
            part_3[k] = np.sum(y, axis=0)
        part_3 = part_3 * means_
        
        part_4 = np.dot(self.resp.T, self.selected) * means_ * digamma(means_)

        self.gamma_vi = self.gamma + part_1 - part_2 + part_3 - part_4
        
    def _estimate_beta(self, X):
        
        means_rj = self.iota_vi / self.iota_vi.sum()

        part_1 = ((1-self.selected) * means_rj * digamma(means_rj.sum())).sum(axis=0)

        y = X + means_rj
        part_2 = ((1-self.selected) * means_rj * digamma(y.sum(axis=1))[:,np.newaxis]).sum(axis=0)

        part_3 = ((1-self.selected) * means_rj * digamma(y)).sum(axis=0)

        part_4 = ((1-self.selected) * means_rj * digamma(means_rj)).sum(axis=0)

        self.iota_vi = self.iota + part_1 - part_2 + part_3 - part_4
        
    def _estimate_log_weights(self):
        
        digamma_sum = digamma(self.weight_concentration_[0] + 
                              self.weight_concentration_[1])
        digamma_a = digamma(self.weight_concentration_[0])
        digamma_b = digamma(self.weight_concentration_[1])
        
        return (digamma_a - digamma_sum + 
                np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
    
    def _estimate_log_prob(self, X):
        
        n_samples, n_features = X.shape
        means_ = self.gamma_vi / self.gamma_vi.sum(axis=1)[:,np.newaxis]
        log_means_ = np.log(means_)
        
        sum_1_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            sum_1_1[:, k] = np.sum(y, axis=1) 
        part_1 = gammaln(means_.sum(axis=1)) - gammaln(sum_1_1)
        part_1 = self.selected.sum(axis=1)[:, np.newaxis] * part_1
        
        
        sum_2_1 = means_ * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_) * digamma(means_.sum(axis=1))[:, np.newaxis]
        sum_2_1 = np.dot(self.selected, sum_2_1.T)
        sum_2_2 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            sum_2_2[:, k] = np.sum(y, axis=1)
        sum_2_2 = digamma(sum_2_2) * np.dot(self.selected, (means_ * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)).T)
        part_2 = sum_2_1 - sum_2_2
        
        
        sum_3_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            y = self.selected * gammaln(y)
            sum_3_1[:, k] = np.sum(y, axis=1)
        sum_3_2 = np.dot(self.selected, gammaln(means_).T)
        part_3 = sum_3_1 - sum_3_2
        
        
        sum_4_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            y = self.selected * digamma(y) * means_[k] * (digamma(self.gamma_vi)[k] - digamma(self.gamma_vi.sum(axis=1))[k] - log_means_[k])
            sum_4_1[:, k] = np.sum(y, axis=1)
        sum_4_2 = np.dot(self.selected, (means_ * digamma(means_) * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)).T)
        part_4 = sum_4_1 - sum_4_2
        
        """
        X_fact = np.empty((n_samples, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                X_fact[i][j] = self.selected[i][j] * np.log(1/(math.factorial(X[i][j])) + 1e-6)
        """
        #return part_1 + part_2 + part_3 + part_4 + X_fact.sum(axis=1)[:, np.newaxis]
        return part_1 + part_2 + part_3 + part_4
    
    def _estimate_weighted_log_prob(self, X):
        
        return self._estimate_log_prob(X) + self._estimate_log_weights()
    
    def _estimate_log_prob_resp(self, X):
        
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis = 1)
        with np.errstate(under = 'ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    
    def _estimate_log_prob_selected(self, X):
        
        n_samples, n_features = X.shape
        
        means_ = self.gamma_vi / self.gamma_vi.sum(axis=1)[:,np.newaxis]
        log_means_ = np.log(means_)

        sum_1_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            sum_1_1[:, k] = np.sum(y, axis=1) 
        part_1 = (self.resp * gammaln(means_.sum(axis=1)) - self.resp * gammaln(sum_1_1)).sum(axis=1)
        #part_1 = part_1 / part_1.sum()
        
        
        sum_2_1 = means_ * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_) * digamma(means_.sum(axis=1))[:, np.newaxis]
        sum_2_1 = np.dot(self.resp, sum_2_1)

        sum_2_2 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = X + means_[k]
            sum_2_2[:, k] = np.sum(y, axis=1)
        sum_2_2 = np.dot((self.resp * digamma(sum_2_2)), (means_ * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)))
        part_2 = sum_2_1 - sum_2_2
        
        
        sum_3_1 = np.empty((self.n_components, n_samples, n_features))
        for k in range(self.n_components):
            y = X + means_[k]
            y = self.resp[:,k][:, np.newaxis] * gammaln(y)
            sum_3_1[k] = y
        sum_3_1 = np.sum(sum_3_1, axis=0)    
        sum_3_2 = np.dot(self.resp, gammaln(means_))
        part_3 = sum_3_1 - sum_3_2
        
        
        sum_4_1 = np.empty((self.n_components, n_samples, n_features))
        for k in range(self.n_components):
            y = X + means_[k]
            y = self.resp[:,k][:, np.newaxis] * digamma(y) * means_[k] * (digamma(self.gamma_vi)[k] - digamma(self.gamma_vi.sum(axis=1))[k] - log_means_[k])
            sum_4_1[k] = y
        sum_4_1 = np.sum(sum_4_1, axis=0)
        sum_4_2 = np.dot(self.resp, (means_ * digamma(means_) * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)))
        part_4 = sum_4_1 - sum_4_2
        
        """
        X_fact = np.empty((n_samples, n_features))
        resp_ = self.resp.sum(axis=1)
        for i in range(n_samples):
            for j in range(n_features):
                X_fact[i][j] = resp_[i] * np.log(1/(math.factorial(X[i][j])) + 1e-6)
        """
        
        #estimate_log_select = part_1[:, np.newaxis] + part_2 + part_3 + part_4 + X_fact
        estimate_log_select = part_2 + part_3 + part_4
        estimate_log_select = estimate_log_select + (digamma(self.xi1) - digamma(self.xi1 + self.xi2))
        return estimate_log_select
    
    def _estimate_log_prob_rejected(self, X):
        
        n_samples, n_features = X.shape
        
        means_ = self.iota_vi / self.iota_vi.sum()
        log_means_ = np.log(means_)

 
        part_1_ = gammaln(means_.sum()) - gammaln((X + means_).sum(axis=1))
        part_1_ = part_1_ / part_1_.sum()
        
        
        sum_2_1_ = means_ * (digamma(self.iota_vi) - digamma(self.iota_vi.sum()) - log_means_) * digamma(means_.sum())
        sum_2_2_ = means_ * (digamma(self.iota_vi) - digamma(self.iota_vi.sum()) - log_means_) * digamma((X + means_).sum(axis=1))[:, np.newaxis]
        part_2_ = sum_2_1_ - sum_2_2_
        
        
        part_3_ = gammaln((X + means_)) - gammaln(means_)


        sum_4_1_ = means_ * (digamma(self.iota_vi) - digamma(self.iota_vi.sum()) - log_means_) * digamma((X + means_))
        sum_4_2_ = means_ * (digamma(self.iota_vi) - digamma(self.iota_vi.sum()) - log_means_) * digamma((means_))
        part_4_ = sum_4_1_ - sum_4_2_
        
        """
        X_fact = np.empty((n_samples, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                X_fact[i][j] = np.log(1/math.factorial(X[i][j]) + 1e-6)
        """
        
        #estimate_log_reject = part_1_[:, np.newaxis] + part_2_ + part_3_ + part_4_ + X_fact
        estimate_log_reject = part_2_ + part_3_ + part_4_
        estimate_log_reject = estimate_log_reject + (digamma(self.xi2) - digamma(self.xi1 + self.xi2))
        return estimate_log_reject
    
    def _estimate_prob_selection(self, X):
        
        selection = self._estimate_log_prob_selected(X)
        rejection = self._estimate_log_prob_rejected(X)
        
        select_exp = np.exp(selection)
        select_exp = np.nan_to_num(select_exp, posinf=1)
        
        reject_exp = np.exp(rejection)
        #reject_exp = np.nan_to_num(reject_exp, posinf=0)
        reject_exp = np.nan_to_num(reject_exp, posinf=1)
        
        #select_exp = sigmoid(selection)
        #reject_exp = sigmoid(rejection)
        
        self.selected = (select_exp + 1e-6) / (select_exp + reject_exp + 1e-6)
        
        return self.selected
        
    def _e_step(self, X):
        
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        prob_selected = self._estimate_prob_selection(X)
        #return np.mean(log_prob_norm), log_resp
        return np.mean(log_prob_norm), log_resp, prob_selected
    
    def _m_step(self, X, log_resp):
        
        n_samples, n_features = X.shape
        
        self.resp = np.exp(log_resp)
        
        self._estimate_weights()
        self._estimate_selections()
        self._estimate_alpha(X)
        self._estimate_beta(X)
        
    def fit_predict(self, X):
        
        n_init = self.n_init
        max_lower_bound = -np.infty 
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        
        clus_update = {}
        sel_update = {}
        
        for init in range(n_init):
            
            self._initialize_parameters(X, random_state)
            lower_bound = -np.infty 
            
            for n_iter in range(1, self.max_iter + 1):
                print(n_iter)
                prev_lower_bound = lower_bound 
                log_prob_norm, log_resp, prob_selected = self._e_step(X)
                self._m_step(X, log_resp)
                clus_update[n_iter] = self.resp
                sel_update[n_iter] = self.selected
                
        _, log_resp, prob_selected = self._e_step(X)
        
        return log_resp, clus_update, prob_selected, sel_update

#------------------------------------------------------------------------------
# Test algorithms
#------------------------------------------------------------------------------ 
from sklearn.metrics.cluster import adjusted_rand_score
import time

tottori_count_raw = pd.read_csv("2000_ASV_nonrarefied.csv", index_col=0)
#tottori_count_raw = pd.read_csv("3000_ASV.csv", index_col=0)
tottori_count = tottori_count_raw.drop(columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus',
       'Species'])
tottori_count = tottori_count.set_index("Row.names")
tottori_count = tottori_count.T
tottori_meta = list(tottori_count.index)
tottori_meta = [i[0] for i in tottori_meta]
tottori_meta = pd.DataFrame(tottori_meta, columns = ["Label"]).reset_index(drop=True)
tottori_meta["Index"] = tottori_count.index
tottori_meta = tottori_meta.set_index("Index")

cdi_count = pd.read_csv("CDI_count_data.csv", index_col=0)
cdi_meta = pd.read_csv("cdi_meta.csv").set_index("sample_id")

ob_count = pd.read_csv("OB_OTU_data.csv", index_col=0)
ob_meta = pd.read_csv("OB_meta_data.csv").set_index("sample_id")

ibd_count = pd.read_csv("IBD_OTU_data.csv", index_col=0)
ibd_meta = pd.read_csv("IBD_meta_data.csv").set_index("sample")

X = check_array(tottori_count, dtype=[np.float64, np.float32])
X = check_array(cdi_count, dtype=[np.float64, np.float32])
X = check_array(ob_count, dtype=[np.float64, np.float32])
X = check_array(ibd_count, dtype=[np.float64, np.float32])

random_state = 42
dmm = DMM_VFS(n_components = 10, max_iter = 100, init_params = "random")

resp, select = dmm._initialize_parameters(X, random_state)

start = time.time()
log_resp_, clus_update, prob_selected, sel_update = dmm.fit_predict(X)
time_computation = time.time() - start 

#resp_ = np.exp(clus_update[1000])
resp_ = clus_update[100]
log_resp_max_ = resp_.argmax(axis=1)

selected_features = prob_selected.sum(axis=0)/prob_selected.shape[0]

df = {'Microbiome_species': tottori_count.columns, 'Selected_probility': selected_features}
clus_selected = pd.DataFrame(data=df).sort_values(by = 'Selected_probility',ascending=False).reset_index(drop=True)
clus_selected.to_csv("Selected_OTUs.csv")

df_cluster = {'Diseases': tottori_meta['Label'], 'Predicted_cluster': log_resp_max_}
clus_labeled = pd.DataFrame(data=df_cluster)
clus_labeled["True_cluster"] = clus_labeled["Diseases"].apply(lambda x: 5 
                                          if x == "C" else 1 
                                          )
clus_labeled = clus_labeled.dropna()
clus_labeled.to_csv("IBD_predicted_clusters.csv")
CDI_predict = pd.read_csv("IBD_predicted_DMM.csv", index_col=0)
CDI_predict["predict"] = CDI_predict["V1"].apply(lambda x: 1 if x == 1 else 0)

ARI_score = adjusted_rand_score(clus_labeled['Predicted_cluster'], clus_labeled['True_cluster'])

#------------------------------------------------------------------------------
# Test name for selected species
#------------------------------------------------------------------------------ 

selected_species_100 = clus_selected[clus_selected["Selected_probility"] > 0.62]

microbiome_species = list(selected_species_100["Microbiome_species"])
microbiome_species_org = microbiome_species
microbiome_species = [i.split(';')[-1][3:] for i in microbiome_species]
microbiome_species = pd.DataFrame(microbiome_species, columns = ["Species"]).reset_index(drop=True)

microbiome_genus = [i.split(';')[-3][3:] for i in microbiome_species_org]
microbiome_genus = pd.DataFrame(microbiome_genus, columns = ["Genus"]).reset_index(drop=True)

microbiome_family = [i.split(';')[-4][3:] for i in microbiome_species_org]
microbiome_family = pd.DataFrame(microbiome_family, columns = ["Family"]).reset_index(drop=True)

microbiome_species_org = pd.DataFrame(microbiome_species_org, columns = ["Full_name"]).reset_index(drop=True)

microbiome_fgs = pd.concat([microbiome_family, microbiome_genus, microbiome_species, 
                                   microbiome_species_org], axis=1)
microbiome_fgs.to_csv("F_G_S_SVVS.csv")

#------------------------------------------------------------------------------
# Test CDI-Health for selected species
#------------------------------------------------------------------------------ 

import matplotlib.pyplot as plt
import seaborn as sns

cdi_meta = pd.read_csv("cdi_meta.csv").set_index("sample_id")
cdi_microbiome = pd.read_csv("cdi_OTUs.csv").set_index("index")

CDI_selected_species_100 = cdi_microbiome[selected_species_100["Microbiome_species"]]

cdi_group = cdi_meta["DiseaseState"]
cdi_group = pd.get_dummies(cdi_group)
drug_group = cdi_meta[["antibiotics >3mo", "protonpump"]]
drug_group = drug_group.rename(columns= {'antibiotics >3mo': 'antibiotics'})
drug_group = pd.get_dummies(drug_group)
cdi_group = cdi_group.rename(columns={'CDI': 'CDI Case', 'ignore-nonCDI': 'Diarrheal Control',
                                      'H': 'Non-Diarrheal Control'})
corr_species_cdi = pd.concat([cdi_group, drug_group, CDI_selected_species_100], axis=1)


plt.subplots(figsize=(10,40))
#corr = corr_function_cdi.corr()
corr = corr_species_cdi.corr(method = "spearman")
corr = corr.drop(index=['CDI Case', 'Non-Diarrheal Control', 'Diarrheal Control', 
                        'antibiotics_no', 'antibiotics_yes', 'protonpump_no', 'protonpump_yes'])
corr = corr[['CDI Case','Non-Diarrheal Control']]
#corr = corr.drop(index=['CDI Case', 'Non-Diarrheal Control', 'Diarrheal Control'])
#corr = corr[['CDI Case','Non-Diarrheal Control', 'Diarrheal Control']]
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm = sns.heatmap(round(corr,2), annot=True, cmap=cmap, fmt=".2f",annot_kws={"size": 20},
                 linewidths=.05)
hm.set_xticklabels(hm.get_xticklabels(), fontsize = 20, rotation=45, horizontalalignment='right')
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize = 20)

otu_cdi = corr.loc[(corr['CDI Case'] > 0.1) & (corr["Non-Diarrheal Control"] < 0)]
otu_cdi = otu_cdi.drop(columns=["Non-Diarrheal Control"])
otu_health = corr.loc[(corr['CDI Case'] < 0) & (corr["Non-Diarrheal Control"] > 0.1)]
otu_health = otu_health.drop(columns=["CDI Case"])
otu_cdi.to_csv("otu_cdi_SVVS.csv")
otu_health.to_csv("otu_health_SVVS.csv")


#------------------------------------------------------------------------------
# Test name for selected species at Tottori 
#------------------------------------------------------------------------------ 

selected_species_100 = clus_selected[clus_selected["Selected_probility"] > 0.9]

species_full_name = tottori_count_raw[['Row.names','Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus',
       'Species']]
species_full_name = species_full_name.set_index("Row.names")
species_full_name = species_full_name.loc[selected_species_100["Microbiome_species"]]
species_full_name.to_csv("F_G_S_SVVS_tottori_100.csv")

#------------------------------------------------------------------------------
# Test Tottori for selected species
#------------------------------------------------------------------------------ 

tottori_abun = tottori_count.divide(tottori_count.sum(axis=1), axis=0)
tott_selected_species_100 = tottori_abun[selected_species_100["Microbiome_species"]]

tott_group = tottori_meta["Label"]
tott_group = pd.get_dummies(tott_group)

corr_species_tott = pd.concat([tott_group, tott_selected_species_100], axis=1)
corr = corr_species_tott.corr(method = "spearman")
corr = corr.drop(index=['C','D'])
corr = corr[['C','D']]

otu_c = corr.loc[(corr['C'] > 0) & (corr['D'] < 0)]
otu_c = otu_c.drop(columns = ['D'])
otu_d = corr.loc[(corr['C'] < 0) & (corr['D'] > 0)]
otu_d = otu_d.drop(columns = ['C'])

otu_c.to_csv("otu_tottori_C_100.csv")
otu_d.to_csv("otu_tottori_D_100.csv")





