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
    "Numerically stable sigmoid function"
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

class DMM_SVVS():
    
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
                #low_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                
                #change = lower_bound - prev_lower_bound
                #if abs(change) < self.tol:
                #    break
                
        _, log_resp, prob_selected = self._e_step(X)
        
        return log_resp, clus_update, prob_selected, sel_update
