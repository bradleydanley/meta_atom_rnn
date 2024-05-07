"""
Purpose: Sequential Deep Learning
Author: Andy
"""


import sys
import os
import pickle
import time

import utils.evaluate as evaluate
import utils.network.train as train
import utils.network.load_results as load_results
from utils.network.support import write_stats
from preprocess_data import preprocess
from utils.general import load_config, parse_args

def experiment(params):

    import time
    # Experiment: Train Algorithm

    if params["experiment"] == 0:
       
        start_time = time.time()

        print("we made it to def experiment() in main.py")
        train.run(params)

        elapsed_time = time.time() - start_time    
        write_stats(params, elapsed_time) 

    # Load in model results, get predictions

    elif params["experiment"] == 1:
        
        import time
        start_time = time.time()

        load_results.run(params)

        elapsed_time = time.time() - start_time
        write_stats(params, elapsed_time)

    # Experiment: Evaluation

    elif params["experiment"] == 2:
        import time

        start_time = time.time()

        evaluate.run(params)

        elapsed_time = time.time() - start_time
        write_stats(params, elapsed_time)

    # Otherwise: Not Implemented
    elif params["experiment"] == 3:
        import time

        start_time = time.time()
        print(f"start time = {start_time}")

        preprocess.run(params) # this is going to launch a kubernetes job.

        elapsed_time = time.time() - start_time
        print(f"elapsed time = {elapsed_time}")
        write_stats(params, elapsed_time)
        

    else:
        raise NotImplementedError


if __name__ == "__main__":

    # Load: Configuration File
    params = load_config(sys.argv)
    params['dataset']['seq_len'] = params['seq_len']
    params['network']['arch'] = params['arch']
   
    print(f"params['dataset']['seq_len'] = {params['dataset']['seq_len']}")
    print(f"params['network']['arch'] = {params['network']['arch']}")
    # Run: Experiment
    experiment(params)
    
