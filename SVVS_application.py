#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:35:15 2022

@author: tungdang
"""

import warnings
from abc import ABCMeta, abstractmethod
from time import time 

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from sklearn.metrics.cluster import adjusted_rand_score
from DMM_Dirichlet_SVVS import DMM_SVVS


dataset_A_count = pd.read_csv("dataset_A_count.csv", index_col=0)
dataset_A_meta = pd.read_csv("dataset_A_meta.csv", index_col=0)

X = check_array(dataset_A_count, dtype=[np.float64, np.float32])

random_state = 42
dmm = DMM_SVVS(n_components = 10, max_iter = 100, init_params = "random")

log_resp_, clus_update, prob_selected, sel_update = dmm.fit_predict(X)
resp_ = clus_update[100]
log_resp_max_ = resp_.argmax(axis=1)

selected_features = prob_selected.sum(axis=0)/prob_selected.shape[0]
df = {'Microbiome_species': dataset_A_count.columns, 'Selected_probility': selected_features}
clus_selected = pd.DataFrame(data=df).sort_values(by = 'Selected_probility',ascending=False).reset_index(drop=True)

df_cluster = {'Diseases': dataset_A_meta['Label'], 'Predicted_cluster': log_resp_max_}
clus_labeled = pd.DataFrame(data=df_cluster)
clus_labeled["True_cluster"] = clus_labeled["Diseases"].apply(lambda x: 1 
                                          if x == "D" else 5)
ARI_score = adjusted_rand_score(clus_labeled["Predicted_cluster"], clus_labeled["True_cluster"])