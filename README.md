# Stochastic variational variable selection - SVVS
[![DOI](https://zenodo.org/badge/387373494.svg)](https://zenodo.org/badge/latestdoi/387373494)

## Getting Started 
### Get the SVVS Source

```
git clone https://github.com/tungtokyo1108/SVVS.git 
```

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

- Import example dataset 
```
dataset_A_count = pd.read_csv("dataset_A_count.csv", index_col=0)
dataset_A_meta = pd.read_csv("dataset_A_meta.csv", index_col=0)
```

- Set up parameters for SVVS function
- - n_components:

```
X = check_array(dataset_A_count, dtype=[np.float64, np.float32])
dmm = DMM_SVVS(n_components = 10, max_iter = 100, init_params = "random")
```




## The published datasets for analyzing 
https://zenodo.org/record/1146764#.Y3cbLuzP2dY

If you have any problem, please contact me via email: dangthanhtung91@vn-bml.com  
