# meta_atom_rnn

Time-series models for learning wavefront propagation with 3x3 metasurface dataset.

# The Dataset

The dataset was generated using [general_3x3](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch). A dataset with 1500 samples is located in the Nautilus hypercluster under the namespace `gpn-missou-muem` in a persistent volume claim called `meep-dataset-v2`. This can be found by logging on to any machine with kubernetes installed and configured for our namespace by running `kubectl get pvc`.

## Folder descriptions

- `k8s` : contains files for data preprocessing, network training, loading results, and evaluation via kubernetes.
  - This folder contains multiple scripts for launching kubernetes jobs.
- `configs` contains the configuration file for the entire pipeline, including a flag `deployment_mode` which gives the user the option to develop locally with a limited dataset, or in the cloud with the full dataset.
- `preprocess_data` contains a script that takes in volumes of DFT electric field data from meep simulations and conditions them for the time series networks here.
- `main.py` is the script we run for all processes if we are developing locally. If we're training locally, `train.sh` automates the experiments.
  - Run `main.py -config configs/params.yaml`. Make sure configs/params.yaml is correct for your preferred deployment_mode and experiment, as well as network training parameters, paths, etc.
- `utils` : contains all our implementations of time series networks using PyTorch lightning, as well as helper functions for data processing, building plots, etc.
