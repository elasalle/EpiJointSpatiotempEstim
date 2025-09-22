# 

This repository contains the code associated to the article *Joint reproduction number and spatial connectivity structure esimtation via sparsity-promoting penalized functional*. 

# Installation

To run our method, follow the instruction below. \
In a terminal, run the following commands : 
- `conda create -n jointEstim python`
- `conda activate jointEstim`
- `pip install jupyterlab matplotlib pandas cvxopt scikit-learn geopandas` 
Finally, go to the `EpiJointSpatiotempEstim` and run the following command: 
`pip install -e .`

# Tutorial 

You can check the notebook `experiments/real_data/europe_africa_ipynb` to see how to use our methods on real data.

# Experiments from the paper. 

## Synthetic data

To reproduce the experiments on synthetic data go into `experiments/synth_w_replications/`. 

To reproduce the data, run `create_data.py`. The data is stored in the `data/` folder.
Each other folder (`blured_true_graph/`, `epiEstim`, `estimated_graph`, `MLE`, `no_graph`, `true_graph`) corresponds to the a given method. Inside each of these folders,  the structure is the same: 
- the `run.py` file to run the method of the synthetic data. 
- the parameters of the method are stored into `parameters.pickle` while the results of the methods are stored into the `expe_res/` folder.
- the `read.py` file allows to read the results and evaluate the method. 
Note: the run and read files must be run from the folder of the chosen method. 

## Real data

To reproduce the experiments on synthetic data go to the tutorial notebook `experiments/real_data/europe_africa_ipynb` that applies our method on a selection of some european and african countries. 