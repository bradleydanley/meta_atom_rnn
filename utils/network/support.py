"""
Purpose: Model Support Tools
Author: Andy
"""


import os
import shutil
import torch
import numpy as np
import csv

from tqdm import tqdm

from utils.network.models import Network
from IPython import embed


def load_model(path, device, params):

    model = Network.load_from_checkpoint(path, params=params).to(device)
    model.eval()

    return model


def get_preds(data, model):

    device = model.device

    all_results = {"truths": [], "preds": []}

    for batch in tqdm(data, desc="Predicting Dataset"):

        samples, labels = batch

        target_seq = labels.size()[1]
        preds = model(samples.to(device), target_seq).detach().cpu()

        all_results["truths"].append(labels)
        all_results["preds"].append(preds)

    all_results["truths"] = torch.stack(all_results["truths"])
    all_results["preds"] = torch.stack(all_results["preds"])

    # n is a result of stacking in line 39,40
    n, b, s, c, h, w = all_results["truths"].size()

    all_results["truths"] = all_results["truths"].view(n * b, s, c, h, w)
    all_results["preds"] = all_results["preds"].view(n * b, s, c, h, w)

    all_results["truths"] = all_results["truths"].numpy()
    all_results["preds"] = all_results["preds"].numpy()

    return all_results


def get_all_model_predictions(params, all_paths, data):

    #path_results = params["paths"]["results"]
    path_results = all_paths['training_root'] 
    device = params["system"]["gpus"]["accelerator"]
    all_versions = params["visualize"]["all_versions"]

    all_results = {}
    for version in all_versions:

        path_folder = os.path.join(path_results, "lightning_logs",
                                   "version_%s" % version, "checkpoints")

        params["network"]["arch"] = version

        if os.path.exists(path_folder):

            path_model = os.path.join(path_folder, os.listdir(path_folder)[-1])

            print("\nPath Model: %s\n" % path_model)

            model = load_model(path_model, device, params)
            results = get_preds(data, model)
            all_results["version_%s" % version] = results

    return all_results


def get_vmin_vmax(preds):

    data = {'version_0': {'real': {'vmins': [], 'vmaxes': []}, 'imag': {'vmins': [], 'vmaxes': []}},
           'version_1': {'real': {'vmins': [], 'vmaxes': []}, 'imag': {'vmins': [], 'vmaxes': []}}}

    print(type(preds))
    print(preds.shape)
    rnn_data = preds['version_0']
    lstm_data = preds['version_1']

    rnn_truths = rnn_data['truths']
    rnn_preds = rnn_data['preds']

    lstm_truths = lstm_data['truths']
    lstm_preds = lstm_data['preds']

    for truth_sample, pred_sample in zip(rnn_truths, rnn_preds):
        real_truth, imag_truth = truth_sample[:,0,:,:], truth_sample[:,1,:,:]
        real_pred, imag_pred = pred_sample[:,0,:,:], pred_sample[:,1,:,:]

        # get min/max for real channel - one min and one max value across truths and preds
        data['version_0']['real']['vmins'].append( min(np.min(real_pred), np.min(real_truth)) )
        data['version_0']['real']['vmaxes'].append( max(np.max(real_pred), np.max(real_truth)) )

        # get min/max for imaginary channel
        data['version_0']['imag']['vmins'].append( min(np.min(imag_pred), np.min(imag_truth)) )
        data['version_0']['imag']['vmaxes'].append( max(np.max(imag_pred), np.max(imag_truth)) )

    for truth_sample, pred_sample in zip(lstm_truths, lstm_preds):
        real_truth, imag_truth = truth_sample[:,0,:,:], truth_sample[:,1,:,:]
        real_pred, imag_pred = pred_sample[:,0,:,:], pred_sample[:,1,:,:]

        # get min/max for real channel - one min and one max value across truths and preds
        data['version_1']['real']['vmins'].append( min(np.min(real_pred), np.min(real_truth)) )
        data['version_1']['real']['vmaxes'].append( max(np.max(real_pred), np.max(real_truth)) )

        # get min/max for imaginary channel
        data['version_1']['imag']['vmins'].append( min(np.min(imag_pred), np.min(imag_truth)) )
        data['version_1']['imag']['vmaxes'].append( max(np.max(imag_pred), np.max(imag_truth)) )

    return data

def move_files(f, folder, dir_path):

    src_file = os.path.join(folder, f)
    dst_file = os.path.join(dir_path, f)
    shutil.copy(src_file, dst_file)
    

def organize_analysis(params, dir_name, tag, file_type):

    sequences = params['visualize']['sequences']
    root = params['paths']['analysis']

    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path) 

    for sequence in sequences:

        folder = os.path.join(root, f'exp_{str(sequence).zfill(2)}/{tag}')
        
        for f in os.listdir(folder):

            if tag == 'loss':

                if f.endswith(file_type) and "epoch" in f:
                    move_files(f, folder, dir_path)

            elif f.endswith(file_type):

                move_files(f, folder, dir_path) 

def write_stats(params, time):

    #path_timing = params['paths']['timing'] 
    path_timing = params['kube']['pp_job']['paths']['timing'] 

    if params['experiment'] == 0:
        label = 'train_network'
    elif params['experiment'] == 1:
        label = 'load_results'
    elif params['experiment'] == 2:
        label = 'run_eval'
    elif params['experiment'] == 3:
        label = 'preprocess'

    path_write = os.path.join(path_timing, f"timing-stats-{label}.csv")
    file_exists = os.path.isfile(path_write)
    
    seq_len = params['dataset']['seq_len']
    network = 'rnn' if params['network']['arch'] == 0 else 'lstm'

    col_names = ['network','Seq length','Time (s)']

    with open(path_write, mode='a') as f:

        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(col_names)
        writer.writerow([network, seq_len, time])
            
