"""
Purpose: Sequential Deep Learning
Author: Andy
"""


import sys
import os
import pickle
import time
from IPython import embed

import utils.evaluate as evaluate
import utils.network.train as train
import utils.network.load_results as load_results
from utils.network.support import write_stats

from utils.general import load_config, parse_args

def experiment(params):

    import time
    # Experiment: Train Algorithm

    if params["experiment"] == 0:
       
        start_time = time.time()

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

    else:
        raise NotImplementedError


if __name__ == "__main__":

    # Load: Configuration File

    params = load_config(sys.argv)
    params['dataset']['seq_len'] = params['seq_len']
    params['network']['arch'] = params['arch']
    print(params['experiment'])
    
    # Run: Experiment
    experiment(params)
    
