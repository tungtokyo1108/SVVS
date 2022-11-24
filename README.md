# Stochastic variational variable selection - SVVS
[![DOI](https://zenodo.org/badge/387373494.svg)](https://zenodo.org/badge/latestdoi/387373494)

## Getting Started 
### Get the SVVS Source

```
git clone https://github.com/tungtokyo1108/SVVS.git 
```

- Install Anaconda and Rstudio in your PC. 
    - https://www.anaconda.com/products/distribution
    - https://posit.co/download/rstudio-desktop/

### How To Use 

- Import packages 
```
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
```

- Import example dataset in  ```data/ ```
```
dataset_A_count = pd.read_csv("datasetA_count.csv", index_col=0)
dataset_A_meta = pd.read_csv("datasetA_meta.csv", index_col=0)
```

- Set up parameters for SVVS function
  - n_components: the maximum number of clusters. Depending on the data, SVVS can decide the number of effective cluster. 
  - max_iter: the maximum number of iterations to perform.
  - init_params: the method used to initialize the weights. There are two options: "kmeans": responsibilities are initialized using kmeans; "random": responsibilities are initialized randomly. Default = "random".
  - weight_concentration_prior: the dirichlet concentration of each component on weight distribution. Default = 0.1.
  - select_prior: the prior on the selection distribution. Default = 1. 

```
X = check_array(dataset_A_count, dtype=[np.float64, np.float32])
dmm = DMM_SVVS(n_components = 10, max_iter = 100, init_params = "random")
```

- Run SVVS 
```
log_resp_, clus_update, prob_selected, sel_update = dmm.fit_predict(X)
```

- Evaluate the number of cluster 
```
resp_ = clus_update[100]
log_resp_max_ = resp_.argmax(axis=1)
df_cluster = {'Diseases': dataset_A_meta['Label'], 'Predicted_cluster': log_resp_max_}
clus_labeled = pd.DataFrame(data=df_cluster)
clus_labeled["True_cluster"] = clus_labeled["Diseases"].apply(lambda x: 2 
                                          if x == "D" else 4)
ARI_score = adjusted_rand_score(clus_labeled["Predicted_cluster"], clus_labeled["True_cluster"])
```

- Evaluate the selected microbiome species 
```
selected_features = prob_selected.sum(axis=0)/prob_selected.shape[0]
df = {'Microbiome_species': dataset_A_count.columns, 'Selected_probility': selected_features}
clus_selected = pd.DataFrame(data=df).sort_values(by = 'Selected_probility',ascending=False).reset_index(drop=True)
``` 

## Directory structure

### Data

### Source code

All of the code is in the ```src/``` folder:

- DMM_Dirichlet_SVVS.py: file contains Python codes for SVVS algorithm
- SVVS_application.py: file that is used to run SVVS algorithm for a sepecific dataset. Outputs of this file are Table S1 and S2. 
- Phylogenetic_analysis.R: file that is used to make phylogenetic analysis for each dataset. Outputs of this file are Figures 2, S4, S5. 

## Get the human gut microbiome datasets

We used human gut microbiome datasets in Duvallet et al. (2017): http://dx.doi.org/10.1038/s41467-017-01973-8. Additional information about the datasets are in the MicrobiomeHD github repo https://github.com/cduvallet/microbiomeHD. MicrobiomeHD contains all 28 datasets of human gut microbiome studies in health and disease. In our study, we used only 3 datasets: cdi_schubert, ibd_gevers and ob_goodrich.

If you have any problem, please contact me via email: dangthanhtung91@vn-bml.com  
