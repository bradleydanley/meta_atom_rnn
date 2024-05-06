## The data has been reduced to volumes in general_3x3/preprocess_data.py
## In this step, we condition the data for the LSTM/RNN experiments. The data will be split into 
##     train/valid in utils/data.py just before training.

## Specifically, we're just looking at the y component of the 1550 wavelength, both real and imaginary components of the dft field

import numpy as np
import pickle
import os
import re
import yaml
import torch
import sys
from IPython import embed

from preprocess_data import marshall_ders, interpolate

def create_folder(folder_path):

    if not os.path.exists(folder_path):

        os.makedirs(folder_path)
        print(f"folder created at {folder_path}.")

    else:

        print(f"folder at {folder_path} already exists.") 


def run(params):

    library = pickle.load(open(params['kube']['paths']['library'],'rb'))

    path_volumes = params['kube']['paths']['data']['volumes']
    path_pp_data = params['kube']['paths']['data']['preprocessed_data']

    #exclude = ['0000.pkl','0001.pkl','0002.pkl','0003.pkl','0004.pkl']
    exclude = []
    i = 0
    #from IPython import embed; embed(); exit()
    with os.scandir(path_volumes) as entries:

        for entry in entries:

            i += 1
            if i > 4:
                break

            if entry.name.endswith(".pkl") and entry.name not in exclude:

                # get radii/phase for sample
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

                # reduced_data has pkl files with phase_vol and amp_vol. -- NOT amp and vol - real and im kept separate
                # reduced_data_cfields has pkl files with complex fields.

                filename = str(idx).zfill(4) + ".pkl"
                filepath = os.path.join(path_pp_data,filename)
    
                data = {'LPA phases': phases, 'data': vol }
                create_folder(path_pp_data)

                with open(filepath,"wb") as f:
                    pickle.dump(data, f)
                    print(f"{filename} dumped to {filepath}", flush=True)

