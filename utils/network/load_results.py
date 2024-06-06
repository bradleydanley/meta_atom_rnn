import os
import yaml
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

from utils.data import load_data
from utils.general import create_folder
from utils.measures import get_all_measures
from utils.network.support import get_all_model_predictions
from utils.network.support import get_vmin_vmax

def get_save_folders(params,path_results,create):
  
    if params['deployment_mode'] == 0:
        path_analysis = params['mounted_paths']['results']['analysis']
    elif params['deployment_mode'] == 1:
        path_analysis = params['kube']['train_job']['paths']['results']['analysis']
    else:
        raise NotImplementedError

    seq_len = str(params['dataset']['seq_len']).zfill(2)
    #Defines a new root for each experiment  
    path_eval_root = os.path.join(path_analysis, "exp_" + seq_len)
    path_loss = os.path.join(path_eval_root, "loss")
    path_images = os.path.join(path_eval_root, "images")
    path_measures = os.path.join(path_eval_root, "measures")
    path_flipbooks = os.path.join(path_eval_root, "flipbooks")
    if params['deployment_mode'] == 0:
        path_results = os.path.join(path_results, "k_" + str(params['dataset']['seq_len']).zfill(2))
    elif params['deployment_mode'] == 1:
        path_results = os.path.join(path_results, "checkpoints", "k_" + str(params['dataset']['seq_len']).zfill(2))
        print("path result =", path_results)
    if create==True:
        create_folder(path_loss)
        create_folder(path_images)
        create_folder(path_measures)
        create_folder(path_flipbooks)

        datasets = ['train', 'valid']
        versions = params['visualize']['all_versions'] 
        #versions = ['version_0', 'version_1']

        for dataset in datasets:
            create_folder(os.path.join(path_measures,dataset))
            for v_tag in versions:
                create_folder(os.path.join(path_images,v_tag))
                create_folder(os.path.join(path_images,v_tag,dataset))       
                create_folder(os.path.join(path_measures,dataset,v_tag)) 

    return {"loss": path_loss,
            "training_root": path_results,
            "eval_root": path_eval_root,
            "images": path_images,
            "measures": path_measures,
            "flipbooks": path_flipbooks}

def save_params(params,all_paths):

    with open(params['paths']['params'], 'r') as file:

        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e} in eval")

    dump_location = os.path.join(all_paths['eval_root'], "params.yaml")

    with open(dump_location, 'w') as file:

        yaml.dump(params,file)

def organize_results(params,path_results,override_seq_len):

    all_versions = params["visualize"]["all_versions"]
    exclude_group = params["visualize"]["exclude_group"]

    # Create: Save Folders

    all_paths = get_save_folders(params,path_results,create=True)

    # Save params:

    #save_params(params,all_paths)

    print("------ Gathering Loss Information ------")
    
    all_loss = []
    for version in all_versions:
        #tag = "version_%s" % version
        tag = version
        print(f"tag = {tag}")
        path_folder = os.path.join(all_paths["training_root"], "lightning_logs", tag)
        print(f"path_folder = {path_folder}")
        path_file = os.path.join(path_folder, "metrics.csv")
        if os.path.exists(path_file):
    
            print("Path Loss: %s\n" % path_file)
    
            data = pd.read_csv(path_file)
    
            all_loss.append(data)

    # Get model Predictions:

    print("------ Gathering Model Predictions ------")

    train, valid = load_data(params,override_seq_len)

    train_preds = get_all_model_predictions(params, all_paths, train)
    valid_preds = get_all_model_predictions(params, all_paths, valid)

    train_vmin_vmax = get_vmin_vmax(train_preds)
    valid_vmin_vmax = get_vmin_vmax(valid_preds)
 
    print("\n------ Scoring Measures ------\n")
    
    all_measures_train = get_all_measures(all_versions, train_preds)
    all_measures_valid = get_all_measures(all_versions, valid_preds)

    #preds = {'train_preds': train_preds, 'valid_preds': valid_preds, }
    preds = {'train': {'preds': train_preds, 'vmin_vmax': train_vmin_vmax},
             'valid': {'preds': valid_preds, 'vmin_vmax': valid_vmin_vmax},
            }
    measures = {'train': {'meas': all_measures_train},
                'valid': {'meas': all_measures_valid},
               }
    return preds, measures, all_loss, all_paths 


def run(params):

    if params['deployment_mode'] == 0:
        path_results = params['mounted_paths']['results']['checkpoints']
        path_analysis = params['mounted_paths']['results']['analysis']
        create_folder(path_analysis)
    elif params['deployment_mode'] == 1:
        path_results =  params['kube']['train_job']['paths']['results']['model_results']
        path_analysis = params['kube']['train_job']['paths']['results']['analysis']
        create_folder(path_analysis)
    
    #create_folder(os.path.join(params['paths']['results'], 'analysis'))

    sequence = params['seq_len']

    #all_preds, all_measures, all_loss = {}, {}, {}

    params['dataset']['seq_len'] = sequence
    
    preds, measures, loss, all_paths = organize_results(params, path_results, sequence)

    #all_loss[sequence] = loss
    #all_preds[sequence] = preds
    #all_measures[sequence] = measures
        
    try:
        with open(os.path.join(path_analysis, 'all_preds_k{:02d}.pkl'.format(sequence)), 'wb') as f: 
            pickle.dump(preds, f)
        print("Preds file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)

    try:
        with open(os.path.join(path_analysis, 'all_measures_k{:02d}.pkl'.format(sequence)), 'wb') as f: 
            pickle.dump(measures, f)
        print("Measures file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)

    try:
        with open(os.path.join(path_analysis, 'all_loss_k{:02d}.pkl'.format(sequence)), 'wb') as f: 
            pickle.dump(loss, f)
        print("Loss file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)
    end_time = datetime.now()
    print(f"Program ended at {end_time}")
