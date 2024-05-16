"""
Purpose: Visualization Library
Author: Andy
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from utils.measures import calc_mse, calc_mae, calc_emd
from utils.network.load_results import get_save_folders

from IPython import embed
plt.style.use("ggplot")

fontsize = 30

def get_colors(num_colors):

    cmap_viridis = plt.cm.get_cmap('viridis')
    colors = [cmap_viridis(i / num_colors) for i in range(num_colors)]

    return colors

def measures_plotter(params, meas_list, experiment, dataset, version, j, figsize, path_results, ylim=None):

    colors = get_colors(3)

    all_paths = get_save_folders(params,path_results,create=False)
   
    fig, ax = plt.subplots(figsize=figsize) 
    for i, measure in enumerate(meas_list.keys()):
        if measure == 'time':
            continue
        
        num_samples = len(meas_list[measure])
        scaled_meas = [val*100 for val in meas_list[measure]]
        ax.plot(meas_list['time'][:num_samples], scaled_meas, '-o', label=measure.upper(), c=colors[i])

    title = 'Exp %s, %s dataset, %s, \nFrame %s' % (experiment, dataset, version, j)
    ax.set_title(title)
    ax.set_xlabel('Wavefront Slice')
    if (len(meas_list[measure]) == 50):
        ax.set_xticks(np.linspace(0, len(meas_list[measure])-1, 6).astype(int))
    else:
        ax.set_xticks(meas_list['time'][:num_samples])
    ax.set_ylabel('Error (%)')
    ax.set_ylim(ylim)
    ax.legend(loc='upper right')

    fig.tight_layout()
    path_save = os.path.join(all_paths['measures'],dataset,version,f'{j}.png')
    fig.savefig(path_save)
                

def plot_truth_and_pred_measures(params, all_preds, path_results, sample_idx = 0,
                                 figsize=(16,4), fontsize=fontsize, ylim=[0,15]):

    sequences = params['visualize']['sequences']
    domain = 0 if params['visualize']['domain']=='real' else 1
   
    for experiment, measures in all_preds.items():
        
        if experiment not in sequences:
            continue
        
        params['dataset']['seq_len'] = experiment
        all_paths = get_save_folders(params,path_results,create=False)
        
        for dataset in measures.keys():
            
            temp_preds = measures[dataset]['preds'] 

            for version in temp_preds.keys():
                
                truths = temp_preds[version]['truths'][sample_idx,:,domain,:,:]
                preds = temp_preds[version]['preds'][sample_idx,:,domain,:,:]

                x_vals = list(range(len(truths)))
                meas_list = {"mse" : [], "mae": [], "emd": [], "time": x_vals}
                
                for j in range(len(truths)):

                    meas_list['mse'].append(calc_mse(preds[j], truths[j]))
                    meas_list['mae'].append(calc_mae(preds[j], truths[j]))
                    meas_list['emd'].append(calc_emd(preds[j], truths[j]))  
                    print(f"plotting measures for {experiment},{dataset},{version},frame{j}")
                    measures_plotter(params,meas_list,experiment,dataset,version,j,figsize,path_results)

                    plt.close()

def get_regression(x, y):

    slope, intercept = np.polyfit(x, y, 1)
    x = np.asarray(x)
    regression_line = slope * x + intercept

    return regression_line

def scatter_plots(params, all_measures, path_results, path_analysis):
   
    colors = get_colors(3)

    all_paths = get_save_folders(params,path_results,create=False)

    all_versions = params['visualize']['all_versions']
    domain = params['visualize']['domain']
    sequences = params['visualize']['sequences']
    datasets = ['train','valid']

    y_train = []
    y_valid = []
    x_labels = []

    for decimation, data in all_measures.items():

        x_labels.append(decimation)
        y_train.append(data['train']['meas'][domain]) 
        y_valid.append(data['valid']['meas'][domain]) 

    for version in all_versions:

        fig, ax = plt.subplots(1, 2, figsize=(10,3))
            
        model = version

        ## -- Plot train data -- ##
        ax[0].scatter(x_labels, [meas[model]['mse']*100 for meas in y_train], label='MSE', color=colors[0])
        ax[0].scatter(x_labels, [meas[model]['mae']*100 for meas in y_train], label='MAE', color=colors[1])
        ax[0].scatter(x_labels, [meas[model]['emd']*100 for meas in y_train], label='EMD', color=colors[2])
        ax[0].set_title(f"Error by decimation - {model.upper()}\n train dataset, {domain} values", fontsize=10)

        # plot regression lines
        for i, metric in enumerate(['mse', 'mae', 'emd']):
            ax[0].plot(x_labels, get_regression(x_labels, [meas[model][metric]*100 for meas in y_train]), color=colors[i])


        ## -- Plot valid data -- ##
        ax[1].scatter(x_labels, [meas[model]['mse']*100 for meas in y_valid], label='MSE', color=colors[0])
        ax[1].scatter(x_labels, [meas[model]['mae']*100 for meas in y_valid], label='MAE', color=colors[1])
        ax[1].scatter(x_labels, [meas[model]['emd']*100 for meas in y_valid], label='EMD', color=colors[2])
        ax[1].set_title(f"Error by decimation - {model.upper()}\nvalid dataset, {domain} values", fontsize=10)

        for i, metric in enumerate(['mse', 'mae', 'emd']):
            ax[1].plot(x_labels, get_regression(x_labels, [meas[model][metric]*100 for meas in y_valid]), color=colors[i])


        for axis in ax:
            axis.legend(loc='upper left', fontsize=fontsize-20)
            axis.set_xlabel("Subset size", fontsize=fontsize-10)
            axis.set_xticks(list(range(0, 61, 5)))
            axis.set_ylabel("Mean error (%)", fontsize=fontsize-10)
            axis.tick_params(axis='y', labelsize=fontsize-15)
            axis.tick_params(axis='x', labelsize=fontsize-15)
            axis.set_ylim([-0.1,15])

        #if model == 'lstm':
        #    ax[0].set_ylim([0,1])

        fig.tight_layout()
        path_save = os.path.join(path_analysis)
        fig.savefig(os.path.join(path_save, f"scatter_{model}.pdf"))

        plt.close()

def plot_bars(params, all_measures, path_results, width=0.08, figsize=(14, 8), fontsize=fontsize):

    domain = params['visualize']['domain']
    sequences = params['visualize']['sequences']
    colors = get_colors(2)

    for experiment, measures in all_measures.items():

        if experiment not in sequences:
            continue

        for key, item in measures.items(): # key = train/valid, measures.items() = {key}_meas, {key}_vmin_vmax

            params['dataset']['seq_len'] = experiment
        
            all_paths = get_save_folders(params,path_results,create=False)

            # Gather: Model Names

            metrics_names = [key for key in item['meas'][domain]['rnn'].keys()] # mse, mae, emd

            # Gather: Bar Heights

            model_names = [key for key in item['meas'][domain].keys()]

            metrics_data = {}
            for results, meas in item.items():

                if results != 'meas':
                    continue
            
                for current_key in meas.keys():
                    if current_key == domain:

                        for model_name in model_names:
                            model_name = model_name.lower()
                            metrics_data[model_name] = meas[current_key][model_name]

            rnn_metrics = list(metrics_data['rnn'].values()) 
            lstm_metrics = list(metrics_data['lstm'].values())

            fig, ax = plt.subplots(figsize=figsize)

            #color_range = np.arange(len(metrics_data)) + 1
            #colors = plt.cm.viridis(color_range / float(max(color_range)))
                
            num_bars = len(model_names)
            num_vals = len(rnn_metrics)
            pos_offset = 1

            positions = np.array(range(num_vals))*(1.5 * num_bars + 1.5 * pos_offset)

            bar_width = 0.2

            # creating the bar plot
            ax.bar(positions - bar_width, rnn_metrics, edgecolor ='black',
                    width = 0.4, label=model_names[0].upper(), color=colors[0])
            ax.bar(positions + bar_width, lstm_metrics, edgecolor ='black',
                    width = 0.4, label=model_names[1].upper(), color=colors[1])

            
            plt.yscale('log')
            
            #ax.set_title(title, fontsize=fontsize)
            ax.set_ylabel("Performance (log scale)", fontsize=fontsize)

            ax.set_ylim([1e-6, 2])

            ax.set_xticks(positions)
            ax.set_xticklabels([metric_name.upper() for metric_name in metrics_names])
            ax.set_xlabel("Metric", fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-5)
            ax.tick_params(axis='x', labelsize=fontsize-5)

            ax.legend(loc="upper right", fontsize=fontsize-5)

            fig.tight_layout()

            path_save = os.path.join(all_paths['measures'],f"{domain}_bars_{key}_{experiment}.pdf") 
            fig.savefig(path_save)
            plt.close()

def create_flipbook_videos(params, image_type, path_results):

    all_versions = params['visualize']['all_versions']
    domain = params['visualize']['domain']
    sequences = params['visualize']['sequences']
    datasets = ['train','valid']
   
    for sequence in sequences:

        print(f"\nWriting {image_type} video for experiment {sequence}...")
        params['dataset']['seq_len'] = sequence
        
        all_paths = get_save_folders(params,path_results,create=False)
    
        for version in all_versions:

            v_tag = version

            for dataset in datasets: 
                
                model = version 
                if image_type == 'images':
                    path_folder = os.path.join(all_paths[image_type], v_tag, dataset)
                    path_save = os.path.join(all_paths['flipbooks'], f'fields_{model}_{dataset}_{sequence}.avi') 
                elif image_type == 'measures':
                    path_folder = os.path.join(all_paths[image_type], dataset, v_tag)
                    path_save = os.path.join(all_paths['flipbooks'], f'meas_{model}_{dataset}_{sequence}.avi') 
                video_helper(path_folder, path_save, image_type, sequence)


def video_helper(path_folder, path_save, image_type, sequence, fps=1):

    if sequence == 50:
        fps = 5
    files_list = os.listdir(path_folder) 

    if image_type == 'images':
        files_sorted = sorted(files_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
    elif image_type == 'measures':
        files_sorted = sorted(files_list, key=lambda x: int(x.split('.')[0]))

    all_frames = [os.path.join(path_folder, f) for f in files_sorted]

    height, width = cv2.imread(all_frames[0]).shape[:2]
    size = (width, height)

    writer = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(path_save, writer, fps, size)

    desc = "Writing %s video %s" % (image_type, len(all_frames))
    with tqdm(total=len(all_frames), desc=desc) as pbar:
        for frame in all_frames:
            video.write(cv2.imread(frame))
            pbar.update(1)

    video.release()

# KEEP #
def image_plotter(images, dataset, titles, version, bounds, path, figsize, fontsize):

    vmin = bounds['min']
    vmax = bounds['max']
    
    #images[0].shape = images[1].shape = images[2].shape = (5, 166, 166) 5 frames
    # titles = ['real truth', 'real pred', 'abs diff']
    
    num_frames = len(images[0])

    for frame_idx in range(num_frames):

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        model = "RNN" if version == 0 else "LSTM"
        fig.suptitle(f"{dataset} Sample - {model}\nFrame {frame_idx}")
             
        for ax, title, im in zip(axs, titles, images):
 
            ax.imshow(im[frame_idx], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(title,fontsize=fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    
            scalar_mappable = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            scalar_mappable.set_array([])  # dummy empty array

            cbar = fig.colorbar(scalar_mappable, ax=ax, orientation='horizontal', pad = 0.05)
                
            #plt.tight_layout()
        path_save = os.path.join(path, f'{model}_{frame_idx}.png')
        fig.savefig(path_save)
            
        plt.close()


def get_image_data(params, preds, domain, dataset, mins, maxes, sample_idx):

    #preds.shape = (num_samples, decimations, real/imag, xdim, ydim)

    domain_tag = 0 if domain == 'real' else 1
    truth = preds['truths'][sample_idx,:,domain_tag,:,:]
    pred = preds['preds'][sample_idx,:,domain_tag,:,:]
   
    bounds = {}
    bounds['min'] = mins[sample_idx]
    bounds['max'] = maxes[sample_idx]

    return truth, pred, bounds

# KEEP #    
def plot_truth_and_pred_images(params, all_data, path_results, sample_idx=0, figsize=(10, 5), fontsize=14):
    
    all_versions = params['visualize']['all_versions']
    domain = params['visualize']['domain']
    sequences = params['visualize']['sequences']
    
    for val, all_preds in all_data.items(): # val=exp number 

        #if val > sequences[-1]:
        if val not in sequences:
            continue
        print(f"\nPlotting images for experiment {val}...")
 
        params['dataset']['seq_len'] = val

        all_paths = get_save_folders(params,path_results,create=False)

        train_preds = all_preds['train']['preds']
        train_vmin_vmax = all_preds['train']['vmin_vmax']
        valid_preds = all_preds['valid']['preds']
        valid_vmin_vmax = all_preds['valid']['vmin_vmax']
    
        for version in all_versions:

            v_tag = version
    
            train_mins = train_vmin_vmax[v_tag][domain]['vmins']
            train_maxes = train_vmin_vmax[v_tag][domain]['vmaxes']
            train_truth, train_pred, train_bounds = get_image_data(params, train_preds[v_tag], domain, "train", train_mins, train_maxes, sample_idx)
            train_diff = np.abs(train_truth-train_pred)
            
            valid_mins = valid_vmin_vmax[v_tag][domain]['vmins']
            valid_maxes = valid_vmin_vmax[v_tag][domain]['vmaxes']
            valid_truth, valid_pred, valid_bounds = get_image_data(params, valid_preds[v_tag], domain, "valid", valid_mins, valid_maxes, sample_idx)
            valid_diff = np.abs(valid_truth-valid_pred)
            
            train_images = [train_truth, train_pred, train_diff]
            valid_images = [valid_truth, valid_pred, valid_diff]
            titles = ["%s Truth" % domain.capitalize(), "%s Prediction" % domain.capitalize(), "Absolute Difference"]

            train_path = os.path.join(all_paths['images'],v_tag,'train')
            valid_path = os.path.join(all_paths['images'],v_tag,'valid')

            image_plotter(train_images, "train", titles, version, train_bounds, train_path, figsize, fontsize)
            image_plotter(valid_images, "train", titles, version, valid_bounds, valid_path, figsize, fontsize)


def plot_loss(params, all_data, path_results, y_lim=[0,0.18], figsize=(10, 5), fontsize=fontsize):

    colors = get_colors(2)
    exclude_group = params['visualize']['exclude_group']
    sequences = params['visualize']['sequences']

    for val, all_loss in all_data.items(): 

        if val not in sequences:
            continue

        params['dataset']['seq_len'] = val
        
        all_paths = get_save_folders(params,path_results,create=False)

        for i, name in enumerate(all_loss[0].columns):

            if name in exclude_group:
                continue

            if "lr" in name:
                tag = "Epoch"
            else:
                tag = name.split("_")[-1]

            x = name.replace("_", " ").title()

            title = "%s vs %s" % (x, tag.capitalize())
            y_label = "MSE Loss"
            x_label = "%s" % tag
           
            path_file = os.path.join(all_paths['loss'], name + tag + f"_{val}.pdf")

            fig, ax = plt.subplots(figsize=figsize)

            for i, data in enumerate(all_loss):

                df = data.dropna(subset=[name])

                if "lr" in name:
                    x_vals = list(range(df.shape[0]))
                else:
                    x_vals = df[tag]

                y_vals = df[name]

                if i == 0:
                    label = "RNN"
                elif i == 1:
                    label = "LSTM"
                elif i == 2:
                    label = "convLSTM"
                else:
                    raise NotImplementedError
                ax.plot(x_vals, y_vals, linewidth=5, label=label, c=colors[i])

            ax.set_title("%s" % title)
            ax.set_xlabel("%s" % x_label.capitalize(), fontsize=fontsize)
            ax.set_ylabel("%s" % y_label, fontsize=fontsize)
            ax.legend(loc="upper right", fontsize=fontsize-6)
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)

            if y_lim is not None:
                ax.set_ylim(y_lim)

            fig.tight_layout()
            #print(f"before saving loss: path_file is {path_file}")
            fig.savefig(path_file)

            plt.close()
