# Stochastic variational variable selection - SVVS
[![DOI](https://zenodo.org/badge/387373494.svg)](https://zenodo.org/badge/latestdoi/387373494)

## Getting Started 
### Get the SVVS Source

```
git clone https://github.com/tungtokyo1108/SVVS.git 
```

- Install Anaconda and Rstudio in your or server PC. 
    - https://www.anaconda.com/products/distribution
    - https://posit.co/download/rstudio-desktop/

### How To Use 

- Import packages 
```
import warnings
from abc import ABCMeta, abstractmethod
from time import time 
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

## Get the human gut microbiome datasets

We used human gut microbiome datasets in Duvallet et al. (2017): http://dx.doi.org/10.1038/s41467-017-01973-8. Additional information about the datasets are in the [MicrobiomeHD github repo](https://github.com/cduvallet/microbiomeHD). MicrobiomeHD contains all 28 datasets of human gut microbiome studies in health and disease. In our paper, we used only 3 datasets: cdi_schubert (dataset B), ibd_gevers (dataset C) and ob_goodrich (dataset D).

## Directory structure

### Data

- The main files for dataset A of our paper in the ```data/``` folder include: 
    - datasetA_count.csv: a OTU count table for dataset A 
    - datasetA_meta.csv: metadata that includes a true label of groups for dataset A 
    - datasetA_tree.nwk: phylogenetic tree for dataset A
- Please follow instructions in the [MicrobiomeHD github repo](https://github.com/cduvallet/microbiomeHD) to get OTU tables, metadata and phylogenetic tree for datasets B, C and D 
- Please follow instructions in the [HMP16SData](https://bioconductor.org/packages/release/data/experiment/vignettes/HMP16SData/inst/doc/HMP16SData.html)  to get OTU tables, metadata and phylogenetic tree for dataset E     

### Source code

All of the code is in the ```src/``` folder, you can use to re-make the analyses, figures and tables in the paper:

- DMM_Dirichlet_SVVS.py: file contains Python codes for SVVS algorithm
- SVVS_application.py: file that is used to run SVVS algorithm for a sepecific dataset. Outputs of this file are ARI score for each dataset in Table 3 and a number of selected species in Table S1 and S2.
- Figure_phylogenetic_analysis.R: file that is used to make phylogenetic analysis for each dataset. Output of this file is Figures 2. 
- Figure_NMDS_analysis.R: file that is used to make NMDS figures. Outputs of this file are Figures 1 and 3.


If you have any problem, please contact me via email: dangthanhtung91@vn-bml.com  
