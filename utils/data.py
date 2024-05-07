"""
Purpose: Data Tools
Author: Andy
"""


import os
import torch
import pickle
import numpy as np
from IPython import embed


class Dataset:

    def __init__(self, samples, labels):

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):

        return self.samples[index], self.labels[index]

    def __len__(self):

        return len(self.samples)

#                                                       (seq, channel, xdim, ydim) # pytorch likes this because batch_first=True
def load_pickle_data(path, seq_len, dtype=np.float32, order=(-1, 0, 1, 2)):

    all_samples, all_labels = [], []

    for current_file in os.listdir(path): # loop through pickle files

        current_file = os.path.join(path, current_file)
        data = pickle.load(open(current_file, "rb"))

        # this is how we get many to many 
        # samples = data["data"][:, :, :, :seq_len]
        # labels = data["data"][:, :, :, seq_len:]

        # get k slices, see config
        samples = data["data"][:, :, :, :1]
        total = data["data"].shape[-1] # total num field slices
        indices = np.linspace(1, total-1, seq_len).astype(int) 
        # do logspace (?) for experiments
        labels = data["data"][:, :, :, indices]

        # put everything in order
        samples = samples.permute(order).to(torch.float32)
        labels = labels.permute(order).to(torch.float32)

        all_labels.append(labels)
        all_samples.append(samples)

    return Dataset(all_samples, all_labels)


def load_data(params, override_seq_len = 0):

    #path_train = params["paths"]["train"]
    #path_valid = params["paths"]["valid"]

    path_train = params['kube']['train_job']['paths']['data']['train']
    path_valid = params['kube']['train_job']['paths']['data']['valid']

    if override_seq_len == 0:
        seq_len = params["dataset"]["seq_len"]
    else:
        seq_len = override_seq_len

    batch_size = params["network"]["batch_size"]
    num_workers = params["system"]["num_workers"]

    train = load_pickle_data(path_train, seq_len)
    valid = load_pickle_data(path_valid, seq_len)

    train = torch.utils.data.DataLoader(train, shuffle=True,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        persistent_workers=True)

    valid = torch.utils.data.DataLoader(valid, shuffle=False,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        persistent_workers=True)
    return train, valid
