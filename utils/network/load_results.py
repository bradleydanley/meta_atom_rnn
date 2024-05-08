import os
import yaml
import numpy as np
import pickle
import pandas as pd
from IPython import embed

from utils.data import load_data
from utils.general import create_folder
from utils.measures import get_all_measures
from utils.network.support import get_all_model_predictions
from utils.network.support import get_vmin_vmax

def get_save_folders(params,path_results,create):

    seq_len = str(params['dataset']['seq_len']).zfill(2)
   
    path_eval_root = os.path.join(path_results, "analysis", "exp_" + seq_len)
    path_loss = os.path.join(path_eval_root, "loss")
    path_images = os.path.join(path_eval_root, "images")
    path_measures = os.path.join(path_eval_root, "measures")
    path_flipbooks = os.path.join(path_eval_root, "flipbooks")
    path_results = os.path.join(path_results, "k_" + str(params['dataset']['seq_len']))

    if create==True:
        create_folder(path_loss)
        create_folder(path_images)
        create_folder(path_measures)
        create_folder(path_flipbooks)

        datasets = ['train', 'valid']
        versions = ['version_0', 'version_1']

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

def organize_results(params):

    path_results = params["paths"]["results"]
    all_versions = params["visualize"]["all_versions"]
    exclude_group = params["visualize"]["exclude_group"]

    # Create: Save Folders

    all_paths = get_save_folders(params,path_results,create=True)

    # Save params:

    save_params(params,all_paths)

    print("------ Gathering Loss Information ------")
    
    all_loss = []
    for version in all_versions:
        tag = "version_%s" % version
        path_folder = os.path.join(all_paths["training_root"], "lightning_logs", tag)
        path_file = os.path.join(path_folder, "metrics.csv")
    
        if os.path.exists(path_file):
    
            print("Path Loss: %s\n" % path_file)
    
            data = pd.read_csv(path_file)
    
            all_loss.append(data)

    # Get model Predictions:

    print("------ Gathering Model Predictions ------")

    train, valid = load_data(params)

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

    #path_results = params['paths']['results']
    path_results =  params['kube']['train_job']['paths']['results']['model_results']
    create_folder(os.path.join(path_results, 'analysis'))
    
    #create_folder(os.path.join(params['paths']['results'], 'analysis'))

    sequences = params['visualize']['sequences'] 

    all_preds, all_measures, all_loss = {}, {}, {}
    for val in sequences:

        params['dataset']['seq_len'] = val

        preds, measures, loss, all_paths = organize_results(params)
        all_loss[val] = loss
        all_preds[val] = preds
        all_measures[val] = measures

        print(f"val = {val}", flush=True)
        
    try:
        with open(os.path.join(path_results, 'analysis', 'all_preds.pkl'), 'wb') as f: 
            pickle.dump(all_preds, f)
        print("Preds file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)

    try:
        with open(os.path.join(path_results, 'analysis', 'all_measures.pkl'), 'wb') as f: 
            pickle.dump(all_measures, f)
        print("Measures file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)

    try:
        with open(os.path.join(path_results, 'analysis', 'all_loss.pkl'), 'wb') as f: 
            pickle.dump(all_loss, f)
        print("Loss file dumped successfully.")
    except Exception as e:
        print("Dump error: ", e)

