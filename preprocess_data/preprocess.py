"""
  meta_atom_rnn/preprocess_data/preprocess.py

 - Prior to this step, the data has been reduced to volumes in separate repo: general_3x3/preprocess_data.py.
 - This script takes in the reduced volumes from the prior step (from configs/params.yaml: mounted_paths.data.volumes if deploying locally,
                                                                                           kube.pp_job.paths.data.volumes if deploying kubernetes)
   and outputs preprocessed files (from configs/params.yaml: mounted_paths.data.preprocessed data
                                                             kube.pp_job.paths.data.preprocessed_data if deploying kubernetes)

 - This step conditions the data for the time series networks. Specifically, we're just looking at the y component of the 1550 wavelength,
   both real and imaginary components of the dft field.

 - In the final step, separate_datsets(), the dataset gets split into the trainset and the valid
   set and moved to their respective folders to prepare for the model.


 
"""


import numpy as np
import pickle
import os
import re
import yaml
import torch
import sys
import glob
import subprocess
from IPython import embed


from preprocess_data import marshall_ders, interpolate


# This method only creates a folder at the location `folder_path` if it does not already exist.
def create_folder(folder_path):

    if not os.path.exists(folder_path):

        os.makedirs(folder_path)
        print(f"folder created at {folder_path}.")

    else:

        print(f"folder at {folder_path} already exists.") 

# Our paths are deployment-dependent. This method takes in the params we loaded from configs/params.yaml
# and determines if we are deploying locally or using kubernetes. It returns the correct paths based on
# this information.
def get_paths(params):

    # A pickle file that contains all the meta-atom radii. It is saved in the meta_atom_rnn repo
    # It matches the file of the same name in the general_3x3 repo from which the dataset was generated.

    library = pickle.load(open(params['mounted_paths']['code']['library'],'rb'))

    if params['deployment_mode'] == 0:  # we are using local compute

        path_volumes = params['mounted_paths']['data']['volumes']
        path_pp_data = params['mounted_paths']['data']['preprocessed_data']
        
    elif params['deployment_mode'] == 1: # we are launching kubernetes jobs

        path_volumes = params['kube']['paths']['data']['volumes']
        path_pp_data = params['kube']['paths']['data']['preprocessed_data']

    return library, path_volumes, path_pp_data


# This method loops through the path containing preprocessed data and moves 80% into a folder
# called `train` - this will be training data for the model - and the remaining 20% into a
# folder called `valid` - this will be the validation data for the model.
def separate_datasets(folder_path):

    ext = '*.pkl'

    samples = glob.glob(os.path.join(folder_path, ext))

    num_samples = len(samples)

    if num_samples > 0:
        
        highest_val = max(int(os.path.splitext(os.path.basename(sample))[0]) for sample in samples)

    else:

        highest_val = None

    train_percent = 80
    valid_percent = 20

    create_folder(os.path.join(folder_path,"train"))
    create_folder(os.path.join(folder_path,"valid"))

    for i, sample in enumerate(samples): 

        if i < train_percent:
            destination = os.path.join(folder_path,"train")
        else:
            destination = os.path.join(folder_path,"valid")

        command = ['mv', sample, destination]
        subprocess.run(command)
            

    

    

# The run() function gets called from main.py. It loops through the directory containing the
# reduced volumes, preprocess them, and then dump them out as pickle files. The pickle files 
# contain pillar radii information as well as DFT field data. The radii information is not used
# in meta_atom_rnn, but will likely be used in future iterations of the models.

# In the future it would be good to parallelize this step! It's pretty slow - taking about 10 
# seconds per sample.
def run(params):

    library, path_volumes, path_pp_data = get_paths(params)

    # 'exclude' is a debugging variable - If you already preprocessed some files and want to 
    # exclude them, put them in this list. Generally, you'll leave it empty for deployment.    

    #exclude = ['0430.pkl','0662.pkl','0720.pkl','0922.pkl','1158.pkl']
    exclude = []

    with os.scandir(path_volumes) as entries:

        
        for entry in entries:

            if entry.name.endswith(".pkl") and entry.name not in exclude:

                # get radii/phase for sample
                # Note, we are not actually using this information for the RNN/LSTM, but it will
                # be needed if you move to a geometry prediction framework (which you'll need to do
                # for inverse design)
                match = re.search(r'\d+',entry.name)
                idx = int(match.group())
                radii = torch.from_numpy(np.asarray(library[idx]))
                phases = interpolate.radii_to_phase(radii)
                phases = torch.from_numpy(np.asarray(phases))
                
                # load in data
                sample_path = os.path.join(path_volumes,entry.name)
                sample = pickle.load(open(sample_path,"rb"))

                # just going to look at the y component of 1550 for now.
                vol = torch.from_numpy(sample[1.55][1])  # shape is [2,166,166,63]
                                                         #          [real/im,xdim,ydim,num_slices]

                # preprocessed_data has pkl files with phase_vol and amp_vol. -- NOT amp and vol - real and im kept separate

                filename = str(idx).zfill(4) + ".pkl"
                filepath = os.path.join(path_pp_data,filename)
   
                print(f"{filename},    {filepath}") 
                data = {'LPA phases': phases, 'data': vol }
                create_folder(path_pp_data)

                with open(filepath,"wb") as f:
                    pickle.dump(data, f)
                    print(f"{filename} dumped to {filepath}", flush=True)
    
    separate_datasets(path_pp_data)
