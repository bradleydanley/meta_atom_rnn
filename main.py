"""
meta_atom_rnn/main.py
Author: Andy


The entire pipeline originates from this file and is controlled by `experiment` in configs/params.yaml
 - 0 = preprocess data
            - Takes data from reduced volumes to slices containing y component of 1550 slices 
              plus phases. This step conditions the reduces volumes for the time series networks.
 - 1 = train network
            - Depending on network.arch, this function call trains an RNN (network.arch=0) or an
              LSTM (network.arch=1) 
            - Outputs of model are dumped to lightning model checkpoints (See path in config file)
 - 2 = load results
            - Loads model checkpoints and parses out loss info, and predictions - This step is
              prep for `run evaluation` 
 - 3 = run evaluation
            - Takes in loss info and predictions, builds plots and other analysis visualizations


To run this script:

 - Need to be running the docker container which can be pulled from kovaleskilab/meep:v3_lightning
    https://hub.docker.com/repository/docker/kovaleskilab/meep/general
 - Run: python3 main.py -config configs/params.yaml

"""


import sys
import os
import pickle
import time

import utils.evaluate as evaluate
import utils.network.train as train
import utils.network.load_results as load_results
from preprocess_data import preprocess
from utils.general import load_config, parse_args

def experiment(params):

    # Experiment: Preprocess data - takes reduced volumes and extracts y component of 1550.
    #                             - also determines and stores the radii / phase of the sample's pillars

    if params["experiment"] == 0:

        print(f"\nBeginning preprocessing step using deployment mode {params['deployment_mode']}\n.")

        preprocess.run(params) 


    # Experiment: Train Algorithm
    elif params["experiment"] == 1:

        print(f"Beginning training for {params['network']['num_epochs']} epochs.")

        train.run(params)


    # Load in model results, get predictions
    elif params["experiment"] == 2:
        
        print("Loading results for all experiments...")

        load_results.run(params)


    # Experiment: Run evaluation
    elif params["experiment"] == 3:

        print("Running evaluation...")

        evaluate.run(params)


    # Otherwise: Not Implemented
    else:
        raise NotImplementedError


if __name__ == "__main__":
    print("got here in main.py 83")
    # Load: Configuration File
    params = load_config(sys.argv)
    if params['deployment_mode'] == 1 or params['bash'] == 1:
        params['dataset']['seq_len'] = params['seq_len']
        params['network']['arch'] = params['arch']

    if params['experiment'] == 1: # then we are training   
        print(f"params['dataset']['seq_len'] = {params['dataset']['seq_len']}")
        print(f"params['network']['arch'] = {params['network']['arch']}")
        print(f"params['experiment'] = {params['experiment']}")

    # Run: Experiment
    experiment(params)
    
