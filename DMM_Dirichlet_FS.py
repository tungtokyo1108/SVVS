#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:13:19 2021

@author: tungbioinfo
"""

import warnings
from abc import ABCMeta, abstractmethod
from time import time 

import math
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma, gammaln, logsumexp
from scipy import linalg

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted
from sklearn import cluster
from sklearn.utils.extmath import row_norms

#------------------------------------------------------------------------------
# Help functions 
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

def _estimate_gaussian_covariances_full(resp_gl, resp_mt, X, select, nk, means, reg_covar):
    
    n_components, n_features = means.shape
    resp = np.dot(resp_gl, resp_mt)
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * (select * diff).T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances_tied(resp_gl, resp_mt, X, select, nk, means, reg_covar):
    
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot((nk * means).T, means)
    covariances = avg_X2 - avg_means2
    covariances /= nk.sum(axis = 0)
    covariances.flat[::len(covariances) + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances_diag(resp_gl, resp_mt, X, select, nk, means, reg_covar):
    
    resp = np.dot(resp_gl, resp_mt)
    avg_X2 = np.dot(resp.T, select * X * X) / nk
    avg_means2 = (np.dot(resp.T, select) * means ** 2) / nk
    avg_X_means = means * np.dot(resp.T, select * X) / nk
    
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

def _compute_precision_cholesky(covariances, covariance_type):
    
    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape 
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance, lower=True)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T 
    elif covariance_type == 'tied':
        _, n_features = covariances.shape 
        cov_chol = linalg.cholesky(covariances, lower=True)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T 
    else: 
        precisions_chol = 1. / np.sqrt(covariances)
        #precisions_chol = np.nan_to_num(precisions_chol) + 1e-6
        
    return precisions_chol

def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape 
        log_det_chol = (np.log(
            matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]))
    elif covariance_type == "tied":
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))
    else:
        log_det_chol = np.log(matrix_chol)
    
    return log_det_chol

#------------------------------------------------------------------------------
# Stochastic Variational Inferenece for Join-Cluster mixture models 
#------------------------------------------------------------------------------

class Integrated_SVVS():
    
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior_gl=None,
                 
                 ## Metabolomics - GMM 
                 weight_concentration_prior_mt=None,
                 weights_init_mt=None, means_init_mt=None, precisions_init_mt=None,
                 mean_precision_prior_mt=None, mean_prior_mt=None,
                 degrees_of_freedom_prior_mt=None, covariance_prior_mt=None,
                 
                 ## Microbiome - DMM
                 weight_concentration_prior_mc=None,
                 weights_init_mc=None,
                 alpha=None, beta=None,
                 
                 ## Random state
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
        
        # Global clustering
        self.weight_concentration_prior_gl = weight_concentration_prior_gl
        
        # Metabolomics - GMM 
        self.weight_concentration_prior_mt = weight_concentration_prior_mt
        self.covariance_type = covariance_type 
        self.weights_init_mt = weights_init_mt
        self.means_init_mt = means_init_mt 
        self.precisions_init_mt = precisions_init_mt
        self.mean_precision_prior_mt = mean_precision_prior_mt
        self.mean_prior_mt = mean_prior_mt
        self.degrees_of_freedom_prior_mt = degrees_of_freedom_prior_mt
        self.covariance_prior_mt = covariance_prior_mt
        
        # Microbiome - DMM
        self.weight_concentration_prior_mc = weight_concentration_prior_mc
        self.weights_init_mc = weights_init_mc
        self.alpha = alpha
        self.beta = beta 
        
    def _initialize_parameters(self, X, Y, random_state):
        
        """
        X: Metabolomics dataset
        Y: Microbiome dataset
        """
        
        n_samples_mt, n_features_mt = X.shape
        n_samples_mc, n_features_mc = Y.shape 
        
        self.weight_concentration_prior_gl = 1./self.n_components
        self.weight_concentration_prior_mt = 1./self.n_components
        self.weight_concentration_prior_mc = 1./self.n_components
        
        self.select_prior_mt = 1
        self.select_prior_mc = 1
        
        random_state = check_random_state(self.random_state)
        
        # Global clustering
        self.resp_gl = np.zeros((n_samples_mc, self.n_components))
        
        if self.init_params == "random":
            self.resp_gl = random_state.rand(n_samples_mc, self.n_components)
            self.resp_gl /= self.resp_gl.sum(axis=1)[:, np.newaxis]
        
        # Metabolomics - GMM 
        self.mean_precision_prior_mt = 1.
        self.mean_prior_mt = X.mean(axis=0)
        self.degrees_of_freedom_prior_mt = n_features_mt
        self.covariance_prior_mt = {
            'full': np.atleast_2d(np.cov(X.T)),
            'tied': np.atleast_2d(np.cov(X.T)),
            'diag': np.var(X, axis=0, ddof=1),
            'spherical': np.var(X, axis=0, ddof=1).mean()
        }[self.covariance_type]
        
        self.resp_mt = np.zeros((self.n_components, self.n_components))
        
        if self.init_params == 'random':
            self.resp_mt = random_state.rand(self.n_components, self.n_components)
            self.resp_mt /= self.resp_mt.sum(axis=1)[:, np.newaxis]
            
        select_mt = np.zeros((n_samples_mt, n_features_mt))
        reject_mt = np.zeros((n_samples_mt, n_features_mt))
        for d in range(n_features_mt):
            chois = random_state.rand(n_samples_mt, 2)
            select_mt[:, d] = chois[:,0]
            reject_mt[:, d] = chois[:,1]
        select_norm_mt = select_mt/(select_mt + reject_mt)
        self.selected_mt = select_norm_mt
        
        gl_mt = np.dot(self.resp_gl, self.resp_mt)
        nk = np.dot(gl_mt.T, self.selected_mt) + 10 * np.finfo(self.resp_gl.dtype).eps
        xk = np.dot(gl_mt.T, self.selected_mt * X) / nk
        sk = {"full": _estimate_gaussian_covariances_full,
              "tied": _estimate_gaussian_covariances_tied,
              "diag": _estimate_gaussian_covariances_diag,
              }[self.covariance_type](self.resp_gl, self.resp_mt, X, self.selected_mt, nk, xk, self.reg_covar)
        
        self._estimate_weights_mt()
        self._estimate_selection_mt()
        self._estimate_means_mt(nk, xk)
        self._estimate_wishart_mt(nk, xk, sk)
        self._estimate_means_rj_mt(X)
        self._estimate_wishart_rj_mt(X)
        
        # Microbiome - DMM 
        self.gamma = np.ones((self.n_components, n_features_mc))
        self.delta = np.ones((self.n_components, n_features_mc))
        self.iota = np.ones(n_features_mc)
        self.kappa = np.ones(n_features_mc)
        
        self.resp_mc = np.zeros((self.n_components , self.n_components))
        
        if self.init_params == "random":
            self.resp_mc = random_state.rand(self.n_components, self.n_components)
            self.resp_mc /= self.resp_mc.sum(axis=1)[:, np.newaxis]
            
        select_mc = np.zeros((n_samples_mc, n_features_mc))
        reject_mc = np.zeros((n_samples_mc, n_features_mc))
        for d in range(n_features_mc):
            chois = random_state.rand(n_samples_mc, 2)
            select_mc[:,d] = chois[:,0]
            reject_mc[:,d] = chois[:,1]
        select_norm_mc = select_mc/(select_mc + reject_mc)
        self.selected_mc = select_norm_mc
        
        self.gamma_vi = self.gamma
        self.delta_vi = self.delta 
        self.iota_vi = self.iota 
        self.kappa_vi = self.kappa 
        
        self._estimate_weights_mc()
        self._estimate_selections_mc()
        
        return self.resp_gl, self.resp_mt, self.resp_mc
    
    """
    Metabolomics - GMM 
    Input data: X
    """
    def _estimate_weights_mt(self):
        
        nk = self.resp_mt.sum(axis=0) + 10 * np.finfo(self.resp_mt.dtype).eps 
        
        self.weight_concentration_mt = (
            1. + nk,
            (self.weight_concentration_prior_mt + 
             np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))        
        
    def _estimate_selection_mt(self):
        
        self.xi1_mt = self.select_prior_mt + self.selected_mt.sum(axis=0)
        self.xi2_mt = self.select_prior_mt + (1 - self.selected_mt).sum(axis=0)
        
    def _estimate_means_mt(self, nk, xk):
        
        self.mean_precision_mt = self.mean_precision_prior_mt + nk
        self.means_mt = ((self.mean_precision_prior_mt * self.mean_prior_mt + nk * xk)/self.mean_precision_mt)
    
    def _estimate_means_rj_mt(self, X):
        
        self.mean_precision_rj_mt = self.mean_precision_prior_mt + (1 - self.selected_mt).sum(axis=0)
        self.means_rj_mt = ((self.mean_precision_prior_mt * self.mean_prior_mt + 
                             ((1 - self.selected_mt) * X).sum(axis=0)) / self.mean_precision_rj_mt)
        
    def _estimate_wishart_mt(self, nk, xk, sk):
        
        _, n_features = xk.shape
        
        if self.covariance_type == 'full':
            
            self.degrees_of_freedom_mt = self.degrees_of_freedom_prior_mt + nk 
            
            self.covariances_mt = np.empty((self.n_components, n_features, n_features))
            
            for k in range(self.n_components):
                diff = xk[k] - self.mean_prior_mt
                self.covariances_mt[k] = (self.covariance_prior_mt + nk[k] * sk[k] + 
                                          nk[k] * self.mean_precision_prior_mt/self.mean_precision_mt[k] * np.outer(diff, diff))
                self.covariances_mt[k] = (self.covariances_mt[k] / self.degrees_of_freedom_mt[k]) + 1e-6
                
        elif self.covariance_type == 'tied':
            
            self.degrees_of_freedom_mt = (self.degrees_of_freedom_prior_mt + nk.sum(axis = 0) / self.n_components)
            
            diff = xk - self.mean_prior_mt
            self.covariances_mt = (
                self.covariance_prior_mt + sk * nk.sum(axis = 0) / self.n_components + 
                self.mean_precision_prior_mt / self.n_components * np.dot(
                    ((nk / self.mean_precision_mt) * diff).T, diff))
            
            self.covariances_mt /= self.degrees_of_freedom_mt
            
        elif self.covariance_type == 'diag':
            
            self.degrees_of_freedom_mt = self.degrees_of_freedom_prior_mt + nk
            
            diff = xk - self.mean_prior_mt
            self.covariances_mt = (
                self.covariance_prior_mt + nk * (
                    sk + (self.mean_precision_prior_mt / self.mean_precision_mt) * np.square(diff)))
            self.covariances_mt /= self.degrees_of_freedom_mt
        
        self.precisions_cholesky_mt = _compute_precision_cholesky(self.covariances_mt, self.covariance_type)
        
    def _estimate_wishart_rj_mt(self, X):
        
        n_samples, n_features = X.shape 
        
        if self.covariance_type == 'full':
            
            self.degrees_of_freedom_rj_mt = self.degrees_of_freedom_prior_mt + self.selected_mt.sum(axis = 0)
            
            diff = X - self.means_rj_mt 
            covariances_mt = np.dot((self.selected_mt * diff).T, diff) / self.selected_mt.sum(axis = 0)
            covariances_mt.flat[::n_features + 1] += self.reg_covar
            
            diff_ = self.means_rj_mt - self.mean_prior_mt
            self.covariances_rj_mt = (self.selected_mt.sum(axis=0) * covariances_mt + 
                                      self.selected_mt.sum(axis=0) * self.mean_precision_prior_mt / 
                                      self.mean_precision_rj_mt * np.outer(diff_, diff_))
            self.covariances_rj_mt = (self.covariances_rj_mt / self.degrees_of_freedom_rj_mt) + 1e-6
            cov_chol = linalg.cholesky(self.covariances_rj_mt, lower=True)
            self.precisions_cholesky_rj_mt = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
            
        elif self.covariance_type == 'diag':
            
            self.degrees_of_freedom_rj_mt = self.degrees_of_freedom_prior_mt + (1 - self.selected_mt).sum(axis = 0)
            
            nk = (1 - self.selected_mt).sum(axis = 0)
            avg_X2 = ((1 - self.selected_mt) * X * X).sum(axis = 0) / nk
            avg_means2 = ((1 - self.selected_mt) * self.means_rj_mt ** 2).sum(axis = 0) / nk
            avg_X_means = (self.means_rj_mt * ((1 - self.selected_mt) * X).sum(axis = 0)) / nk
            sk = avg_X2 - 2 * avg_X_means + avg_means2 + self.reg_covar
            
            diff = self.means_rj_mt - self.mean_prior_mt 
            self.covariances_rj_mt = (nk * (sk + (self.mean_precision_prior_mt / self.mean_precision_rj_mt) * np.square(diff)))
            self.covariances_rj_mt /= self.degrees_of_freedom_rj_mt 
            self.precisions_cholesky_rj_mt = 1. / np.sqrt(self.covariances_rj_mt)
            
    def _estimate_log_weights_mt(self):
        
        digamma_sum = digamma(self.weight_concentration_mt[0] + 
                              self.weight_concentration_mt[1])
        digamma_a = digamma(self.weight_concentration_mt[0])
        digamma_b = digamma(self.weight_concentration_mt[1])
        
        return (digamma_a - digamma_sum + 
                np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
    
    def _estimate_log_prob_mt(self, X):
        
        n_samples, n_features = X.shape 
        n_components, _ = self.means_mt.shape 
        
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_mt, self.covariance_type, n_features)
        
        if self.covariance_type == 'full':
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(self.means_mt, self.precisions_cholesky_mt)):
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                log_prob[:, k] = np.sum(np.dot(self.resp_gl.T, (np.square(y) * self.selected_mt)) * self.degrees_of_freedom_mt[k], axis=1)
            
        elif self.covariance_type == 'diag':
            precisions = self.degrees_of_freedom_mt * (self.precisions_cholesky_mt ** 2)
            global_selected_mt = np.dot(self.resp_gl.T, self.selected_mt)
            log_prob = (np.dot(global_selected_mt, (self.means_mt ** 2 * precisions).T) - 
                        2. * np.dot(np.dot(self.resp_gl.T, self.selected_mt * X), (self.means_mt * precisions).T) + 
                        np.dot(np.dot(self.resp_gl.T, self.selected_mt * X ** 2), precisions.T))
            
        log_gauss = (-.5 * (n_features * np.log(2 * np.pi) + log_prob)) + .5 * np.dot(np.dot(self.resp_gl.T, self.selected_mt), log_det.T)
        
        log_lambda = n_features * np.log(2.) + digamma(.5 * (self.degrees_of_freedom_mt - np.arange(0, n_features)))
        log_lambda = .5 * (log_lambda - n_features / self.mean_precision_mt)
        
        return log_gauss + np.dot(np.dot(self.resp_gl.T, self.selected_mt), log_lambda.T)
    
    def _estimate_weighted_log_prob_mt(self, X):
        
        return self._estimate_log_prob_mt(X) + self._estimate_log_weights_mt()
    
    def _estimate_log_prob_resp_mt(self, X):
        
        weighted_log_prob = self._estimate_weighted_log_prob_mt(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis = 1)
        with np.errstate(under = 'ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp 
    
    def _estimate_log_prob_selected_mt(self, X):
        
        n_samples, n_features = X.shape 
        n_components, _ = self.means_mt.shape 
        
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_mt, self.covariance_type, n_features)
        
        if self.covariance_type == 'full':
            log_prob_selected = np.empty((n_samples, n_features))
            for k, (mu, prec_chol) in enumerate(zip(self.means_mt, self.precisions_cholesky_mt)):
                y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
                log_pro_select = np.square(y) * self.degrees_of_freedom_mt[k]
                log_pro_select = (log_pro_select.T * (np.dot(self.resp_gl, self.resp_mt).T)[k]).T
                log_prob_selected += log_pro_select
                
        elif self.covariance_type == 'diag':
            
            precisions = self.degrees_of_freedom_mt * self.precisions_cholesky_mt ** 2
            log_prob_selected = (np.dot(np.dot(self.resp_gl, self.resp_mt), ((self.means_mt ** 2) * precisions)) - 
                                 (2. * X * np.dot(np.dot(self.resp_gl, self.resp_mt), (self.means_mt * precisions))) + 
                                 ((X ** 2) * np.dot(np.dot(self.resp_gl, self.resp_mt), precisions)))
        
        log_gauss_selected = ((.5 * (log_prob_selected)) + .5 * np.dot(np.dot(self.resp_gl, self.resp_mt), log_det))
        
        log_lambda = digamma(.5 * (self.degrees_of_freedom_mt - np.arange(0, n_features))) 
        log_lambda = .5 * (log_lambda - n_features / self.mean_precision_mt)
        
        estimate_log_gauss_selected = log_gauss_selected + np.dot(np.dot(self.resp_gl, self.resp_mt), log_lambda)
        estimate_log_gauss_selected = estimate_log_gauss_selected + (digamma(self.xi1_mt) - digamma(self.xi1_mt + self.xi2_mt))
        
        return estimate_log_gauss_selected
    
    def _estimate_log_prob_rejected_mt(self, X):
        
        n_samples, n_features = X.shape 
        
        if self.covariance_type == 'full':
            log_det = np.log(np.diag(self.precisions_cholesky_rj_mt))
            
            y = np.dot(X, self.precisions_cholesky_rj_mt) - np.dot(self.means_rj_mt, self.precisions_cholesky_rj_mt)
            log_prob_rejected = np.square(y) * self.degrees_of_freedom_rj_mt
            
        elif self.covariance_type == 'diag': 
            log_det = np.log(self.precisions_cholesky_rj_mt)
            
            precisions_rj_ = self.degrees_of_freedom_rj_mt * self.precisions_cholesky_rj_mt ** 2 
            log_prob_rejected = (((self.means_rj_mt ** 2) * precisions_rj_) - 
                                 (2. * X * (self.means_rj_mt * precisions_rj_)) + 
                                 ((X ** 2) * precisions_rj_))
        
        log_gauss_rejected = ((-.5 * (log_prob_rejected)) + .5 * log_det)
        
        log_lambda_rejected = digamma(.5 * (self.degrees_of_freedom_rj_mt - np.arange(0, n_features)))
        log_lambda_rejected = .5 * (log_lambda_rejected - n_features / self.mean_precision_rj_mt)
        
        estimate_log_gauss_rejected = log_gauss_rejected + log_lambda_rejected
        estimate_log_gauss_rejected = estimate_log_gauss_rejected + (digamma(self.xi2_mt) - digamma(self.xi1_mt + self.xi2_mt))
        
        return estimate_log_gauss_rejected 
    
    def _estimate_prob_selection_mt(self, X):
        
        selection = self._estimate_log_prob_selected_mt(X)
        rejection = self._estimate_log_prob_rejected_mt(X)
        
        select_exp = np.exp(selection)
        select_exp = np.nan_to_num(select_exp, posinf=1)
        
        reject_exp = np.exp(rejection)
        reject_exp = np.nan_to_num(reject_exp, posinf=1)
        
        #select_exp = sigmoid(selection)
        #reject_exp = sigmoid(rejection)
        
        self.selected_mt = (select_exp + 1e-6) / (select_exp + reject_exp + 1e-6)
        
        return self.selected_mt
            
    
    """
    Microbiome - DMM 
    Input data: Y
    """
    
    def _estimate_weights_mc(self):
        
        nk = self.resp_mc.sum(axis = 0) + 10 * np.finfo(self.resp_mc.dtype).eps
        
        self.weight_concentration_mc = (
            1. + nk, 
            (self.weight_concentration_prior_mc + 
             np.hstack((np.cumsum(nk[::1])[-2::-1], 0))))
    
    def _estimate_selections_mc(self):
        
        self.xi1_mc = self.select_prior_mc + self.selected_mc.sum(axis = 0)
        self.xi2_mc = self.select_prior_mc + (1 - self.selected_mc).sum(axis = 0)
        
    def _estimate_alpha_mc(self, Y): 
        
        n_samples, n_features = Y.shape 
        means_mc = self.gamma_vi / self.gamma_vi.sum(axis = 1)[:, np.newaxis]
        join_clus_mc = np.dot(self.resp_gl, self.resp_mc)
        
        part_1 = np.dot(join_clus_mc.T, self.selected_mc) * means_mc * digamma(means_mc.sum(axis=1))[:, np.newaxis]
        
        dig_sum_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = Y + means_mc[k]
            dig_sum_1[:,k] = np.sum(y, axis = 1)
        part_2 = np.dot((join_clus_mc * digamma(dig_sum_1)).T, self.selected_mc) * means_mc
        
        part_3 = np.empty((self.n_components, n_features))
        for k in range(self.n_components):
            y = Y * means_mc[k]
            y = self.selected_mc * digamma(y) * join_clus_mc[:,k][:, np.newaxis]
            part_3[k] = np.sum(y, axis = 0)
        part_3 = part_3 * means_mc
        
        part_4 = np.dot(join_clus_mc.T, self.selected_mc) * means_mc * digamma(means_mc)
        
        self.gamma_vi = self.gamma + part_1 - part_2 + part_3 - part_4 
        
    def _estimate_beta_mc(self, Y):
        
        means_rj_mc = self.iota_vi / self.iota_vi.sum()
        
        part_1 = ((1 - self.selected_mc) * means_rj_mc * digamma(means_rj_mc.sum())).sum(axis = 0)
        
        y = Y + means_rj_mc
        part_2 = ((1 - self.selected_mc) * means_rj_mc * digamma(y.sum(axis=1))[:,np.newaxis]).sum(axis=0)
        
        part_3 = ((1 - self.selected_mc) * means_rj_mc * digamma(y)).sum(axis=0)
        
        part_4 = ((1 - self.selected_mc) * means_rj_mc * digamma(means_rj_mc)).sum(axis=0)
        
        self.iota_vi = self.iota + part_1 - part_2 + part_3 - part_4
        
    def _estimate_log_weights_mc(self):
        
        digamma_sum = digamma(self.weight_concentration_mc[0] + 
                              self.weight_concentration_mc[1])
        digamma_a = digamma(self.weight_concentration_mc[0])
        digamma_b = digamma(self.weight_concentration_mc[1])
        
        return (digamma_a - digamma_sum + 
                np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
    
    def _estimate_log_prob_mc(self, Y):
        
        n_samples, n_features = Y.shape 
        means_mc = self.gamma_vi / self.gamma_vi.sum(axis = 1)[:, np.newaxis]
        log_means_mc = np.log(means_mc)
        
        sum_1_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = Y + means_mc[k]
            sum_1_1[:, k] = np.sum(y, axis = 1)
        part_1 = gammaln(means_mc.sum(axis = 1)) - gammaln(sum_1_1)
        part_1 = self.selected_mc.sum(axis = 1)[:, np.newaxis] * part_1
        part_1 = np.dot(self.resp_gl.T, part_1)
        
        
        sum_2_1 = means_mc * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis = 1))[:, np.newaxis] - log_means_mc) * digamma(means_mc.sum(axis = 1))[:, np.newaxis]
        sum_2_1 = np.dot(np.dot(self.resp_gl.T, self.selected_mc), sum_2_1.T)
        sum_2_2 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = Y + means_mc[k]
            sum_2_2[:, k] = np.sum(y, axis = 1)
        sum_2_2 = digamma(sum_2_2) * np.dot(np.dot(self.resp_gl.T, self.selected_mc), (means_mc * (digamma(self.gamma_vi) - digamma(self.gamma_vi.sum(axis = 1))[:, np.newaxis] - log_means_mc)).T)
        part_2 = sum_2_1 - sum_2_2 
        
        
        sum_3_1 = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            y = Y + means_mc[k]
            y = 
        
        
        
    
    














































