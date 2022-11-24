#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:19:24 2021

@author: tungdang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:16:33 2021

@author: tungbioinfo
"""

import warnings
from abc import ABCMeta, abstractmethod
from time import time 

import math
from scipy.misc import factorial
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.special import betaln, digamma, gammaln, logsumexp, polygamma
from scipy import linalg

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn import cluster
from sklearn.utils.extmath import row_norms

#------------------------------------------------------------------------------
# Check gamma + delta update 
#------------------------------------------------------------------------------

gamma_vi = np.ones((n_components, n_features)) 
delta_vi = np.ones((n_components, n_features)) 

n_components = 5
n_samples, n_features = X.shape

nk = np.dot(resp.T, select) + 10 * np.finfo(resp.dtype).eps
gamma = np.ones((n_components, n_features)) 
delta = np.ones((n_components, n_features)) 

means_ = gamma_vi / gamma_vi.sum(axis=1)[:,np.newaxis]

part_1 = np.dot(resp.T, select) * means_ * digamma(means_.sum(axis=1))[:, np.newaxis]

dig_sum_1 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    dig_sum_1[:,k] = np.sum(y, axis=1)
part_2 = np.dot((resp * digamma(dig_sum_1)).T, select) * means_

#dig_sum_1_ = digamma(dig_sum_1)
#diff_1 = digamma(means_.sum(axis=1)) - np.sum(digamma(dig_sum_1), axis=0)
"""
part_ = np.empty((n_components, n_samples, n_features))
for k in range(n_components):
    y = X + means_[k]
    y = select * digamma(y) * resp[:,k][:, np.newaxis]
    part_[k] = y
    
part_3_ = np.sum(part_, axis=1) 
"""

part_3 = np.empty((n_components, n_features))
for k in range(n_components):
    y = X + means_[k]
    y = select * digamma(y) * resp[:,k][:, np.newaxis]
    part_3[k] = np.sum(y, axis=0)
part_3 = part_3 * means_
    
part_4 = np.dot(resp.T, select) * means_ * digamma(means_)

gamma_vi = gamma + part_1 - part_2 + part_3 - part_4
#gamma_vi = gamma + part_1 - part_2 - part_4
      
gamma_dig = digamma(gamma_vi)  
delta_vi = gamma_vi

#------------------------------------------------------------------------------
# Check iota + kappa update 
#------------------------------------------------------------------------------

iota = np.ones((n_features)) 
kappa = np.ones((n_features)) 

iota_vi = np.ones((n_features)) 
kappa_vi = np.ones((n_features)) 

means_rj = iota_vi / iota_vi.sum()

part_1 = ((1-select) * means_rj * digamma(means_rj.sum())).sum(axis=0)

y = X + means_rj
part_2 = ((1-select) * means_rj * digamma(y.sum(axis=1))[:,np.newaxis]).sum(axis=0)

part_3 = ((1-select) * means_rj * digamma(y)).sum(axis=0)

part_4 = ((1-select) * means_rj * digamma(means_rj)).sum(axis=0)

iota_vi = iota + part_1 - part_2 + part_3 - part_4
#iota_vi = iota + part_1 - part_2 - part_4

iota_dig = digamma(iota_vi)
kappa_vi = iota_vi

#------------------------------------------------------------------------------
# Check log DM update 
#------------------------------------------------------------------------------

means_ = gamma_vi / gamma_vi.sum(axis=1)[:,np.newaxis]
log_means_ = np.log(means_)
#log_means_ = np.where(np.isnan(log_means_), ma.array(log_means_, mask=np.isnan(log_means_)).mean(axis=0), log_means_)

sum_1_1 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    sum_1_1[:, k] = np.sum(y, axis=1) 
part_1 = gammaln(means_.sum(axis=1)) - gammaln(sum_1_1)
part_1 = select.sum(axis=1)[:, np.newaxis] * part_1

sum_2_1 = means_ * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_) * digamma(means_.sum(axis=1))[:, np.newaxis]
sum_2_1 = np.dot(select, sum_2_1.T)
sum_2_2 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    sum_2_2[:, k] = np.sum(y, axis=1)
sum_2_2 = digamma(sum_2_2) * np.dot(select, (means_ * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)).T)
part_2 = sum_2_1 - sum_2_2

sum_3_1 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    y = select * gammaln(y)
    sum_3_1[:, k] = np.sum(y, axis=1)
sum_3_2 = np.dot(select, gammaln(means_).T)
part_3 = sum_3_1 - sum_3_2

sum_4_1 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    y = select * digamma(y) * means_[k] * (digamma(gamma_vi)[k] - digamma(gamma_vi.sum(axis=1))[k] - log_means_[k])
    sum_4_1[:, k] = np.sum(y, axis=1)
sum_4_2 = np.dot(select, (means_ * digamma(means_) * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)).T)
part_4 = sum_4_1 - sum_4_2
#part_4 = sum_4_2

X_fact = np.empty((n_samples, n_features))
for i in range(n_samples):
    for j in range(n_features):
        X_fact[i][j] = select[i][j] * np.log(1/(math.factorial(X[i][j])) + 1e-6)

estimate_log_dm = part_1 + part_2 + part_3 + part_4 + X_fact.sum(axis=1)[:, np.newaxis]
#estimate_log_dm = part_2 + part_3 + part_4 + X_fact.sum(axis=1)[:, np.newaxis]

#------------------------------------------------------------------------------
# Check estimate log weight
#------------------------------------------------------------------------------

nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        
weight_concentration_ = (
            1. + nk,
            (weight_concentration_prior + 
             np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))

digamma_sum = digamma(weight_concentration_[0] + 
                              weight_concentration_[1])
digamma_a = digamma(weight_concentration_[0])
digamma_b = digamma(weight_concentration_[1])
        
estimate_log_weight = (digamma_a - digamma_sum + 
                np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))

#------------------------------------------------------------------------------
# Check estimate allocator variables
#------------------------------------------------------------------------------

weighted_log_prob = estimate_log_dm + estimate_log_weight
log_prob_norm = logsumexp(weighted_log_prob, axis = 1)
with np.errstate(under = 'ignore'):
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

resp_update = np.exp(log_resp)
log_resp_max = log_resp.argmax(axis=1)

#------------------------------------------------------------------------------
# Check estimate dirichlet selected 
#------------------------------------------------------------------------------

xi1 = select_prior + select.sum(axis=0)
xi2 = select_prior + (1 - select).sum(axis=0)

#------------------------------------------------------------------------------
# Check log selection update 
#------------------------------------------------------------------------------

means_ = gamma_vi / gamma_vi.sum(axis=1)[:,np.newaxis]
log_means_ = np.log(means_)

sum_1_1 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    sum_1_1[:, k] = np.sum(y, axis=1) 
part_1 = (resp * gammaln(means_.sum(axis=1)) - resp * gammaln(sum_1_1)).sum(axis=1)
part_1 = part_1 / part_1.sum()


sum_2_1 = means_ * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_) * digamma(means_.sum(axis=1))[:, np.newaxis]
sum_2_1 = np.dot(resp, sum_2_1)

sum_2_2 = np.empty((n_samples, n_components))
for k in range(n_components):
    y = X + means_[k]
    sum_2_2[:, k] = np.sum(y, axis=1)
sum_2_2 = np.dot((resp * digamma(sum_2_2)), (means_ * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)))
part_2 = sum_2_1 - sum_2_2


sum_3_1 = np.empty((n_components, n_samples, n_features))
for k in range(n_components):
    y = X + means_[k]
    y = resp[:,k][:, np.newaxis] * gammaln(y)
    sum_3_1[k] = y
sum_3_1 = np.sum(sum_3_1, axis=0)    
sum_3_2 = np.dot(resp, gammaln(means_))
part_3 = sum_3_1 - sum_3_2


sum_4_1 = np.empty((n_components, n_samples, n_features))
for k in range(n_components):
    y = X + means_[k]
    y = resp[:,k][:, np.newaxis] * digamma(y) * means_[k] * (digamma(gamma_vi)[k] - digamma(gamma_vi.sum(axis=1))[k] - log_means_[k])
    sum_4_1[k] = y
sum_4_1 = np.sum(sum_4_1, axis=0)
sum_4_2 = np.dot(resp, (means_ * digamma(means_) * (digamma(gamma_vi) - digamma(gamma_vi.sum(axis=1))[:, np.newaxis] - log_means_)))
part_4 = sum_4_1 - sum_4_2
#part_4 = sum_4_2

X_fact = np.empty((n_samples, n_features))
resp_ = resp.sum(axis=1)
for i in range(n_samples):
    for j in range(n_features):
        X_fact[i][j] = resp_[i] * np.log(1/(math.factorial(X[i][j])) + 1e-6)
        #X_fact[i][j] = 1/(math.factorial(X[i][j]))
        
"""
X_fact_3d = np.empty((n_components, n_samples, n_features))
for k in range(n_components):
    for i in range(n_samples):
        for j in range(n_features):
            X_fact_3d[k][i][j] = resp[i][k] * np.log(1/(math.factorial(X[i][j])) + 1e-6)
            #X_fact[i][j] = 1/(math.factorial(X[i][j]))
X_fact_ = np.sum(X_fact_3d, axis=0)
"""

#estimate_log_select = part_1[:, np.newaxis] + part_2 + part_3 + part_4 + X_fact
estimate_log_select = part_2 + part_3 + part_4 + X_fact
estimate_log_select = estimate_log_select + (digamma(xi1) - digamma(xi1 + xi2))

check = pd.DataFrame(data={"X0":part_1 ,"X1":part_2[:, 2323], "X2":part_3[:, 2323], "X3":part_4[:, 2323]})

#------------------------------------------------------------------------------
# Check log rejection update 
#------------------------------------------------------------------------------

means_ = iota_vi / iota_vi.sum()
log_means_ = np.log(means_)

 
part_1_ = gammaln(means_.sum()) - gammaln((X + means_).sum(axis=1))
part_1_ = part_1_ / part_1_.sum()


sum_2_1_ = means_ * (digamma(iota_vi) - digamma(iota_vi.sum()) - log_means_) * digamma(means_.sum())
sum_2_2_ = means_ * (digamma(iota_vi) - digamma(iota_vi.sum()) - log_means_) * digamma((X + means_).sum(axis=1))[:, np.newaxis]
part_2_ = sum_2_1_ - sum_2_2_


part_3_ = gammaln((X + means_)) - gammaln(means_)


sum_4_1_ = means_ * (digamma(iota_vi) - digamma(iota_vi.sum()) - log_means_) * digamma((X + means_))
sum_4_2_ = means_ * (digamma(iota_vi) - digamma(iota_vi.sum()) - log_means_) * digamma((means_))
part_4_ = sum_4_1_ - sum_4_2_
#part_4 = sum_4_2


X_fact = np.empty((n_samples, n_features))
for i in range(n_samples):
    for j in range(n_features):
        X_fact[i][j] = np.log(1/math.factorial(X[i][j]) + 1e-6)

"""
part3_row_max = np.empty((n_samples))

for i in range(n_samples):
   part3_row_max[i] = np.max(part_3[i])
"""    

#estimate_log_reject = part_1_[:, np.newaxis] + part_2_ + part_3_ + part_4_ + X_fact
estimate_log_reject =  part_2_ + part_3_ + part_4_ + X_fact
estimate_log_reject = estimate_log_reject + (digamma(xi2) - digamma(xi1 + xi2))

#------------------------------------------------------------------------------
# Check estimate selection variables
#------------------------------------------------------------------------------

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
    

select_exp = np.exp(estimate_log_select)
select_exp = np.nan_to_num(select_exp, posinf=1)

select_exp_ = sigmoid(estimate_log_select)
select_ = select_exp.sum(axis=0)/336

reject_exp = np.exp(estimate_log_reject)
reject_exp = np.nan_to_num(reject_exp, posinf=1)

reject_exp_ = sigmoid(estimate_log_reject)
reject_ = reject_exp_.sum(axis=0)/336

select_update = (select_exp + 1e-6) / (select_exp + reject_exp + 1e-6)
select_update_ = select_update.sum(axis=0)/336

select_update_sig = (select_exp_ + 1e-6) / (select_exp_ + reject_exp_ + 1e-6)
select_update_sig_ = select_update_sig.sum(axis=0)/336





















