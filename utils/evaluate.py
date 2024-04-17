"""
Purpose: Evaluate Trained Systems
Author: Andy
"""


import os
import shutil
import yaml
import numpy as np
import pickle
from IPython import embed

#from utils.data import load_data
from utils.general import create_folder
from utils.measures import get_all_measures
from utils.network.support import get_all_model_predictions, organize_analysis
from utils.plots import plot_loss, plot_bars
import utils.plots as plots
from utils.network.load_results import get_save_folders


def run(params):

    # N O T E!! we are running eval for real-valued results.
    print("------ Running Evaluation ------")

    path_analysis = params['paths']['analysis']

    all_measures = pickle.load(open(os.path.join(path_analysis,'all_measures.pkl'),'rb')) # this returns aggregates for bar plots
    all_preds = pickle.load(open(os.path.join(path_analysis,'all_preds.pkl'),'rb'))
    all_loss = pickle.load(open(os.path.join(path_analysis,'all_loss.pkl'),'rb'))

    print("------ Plotting Loss ------")
    #plot_loss(params, all_loss) #pass domain_choice='imag' if you want imaginary vals 

    print("------ Plotting Bar Plots ------")
    plot_bars(params, all_measures)     
    
    print("------ Plotting Images ------\n")

    plots.plot_truth_and_pred_images(params, all_preds) 

    print("------ Creating Images Flipbooks ------\n")
    plots.create_flipbook_videos(params,'images')
    
    print("------ Plotting Measures ------\n")
    plots.plot_truth_and_pred_measures(params, all_preds)

    print("------ Creating Measures Flipbooks ------\n")

    plots.create_flipbook_videos(params,'measures') 
  
    print("----- Evaluation complete. -----")

    # move all presentation files to a single folder #
   
    flipbooks_dir = "all_flipbooks"
    organize_analysis(params, flipbooks_dir, 'flipbooks', '.avi')     

    loss_dir = "all_loss"
    organize_analysis(params, loss_dir, 'loss', '.pdf')     

    bars_dir = "all_bars"
    organize_analysis(params, bars_dir, 'measures', '.pdf')

    print("----- Creating scatter plot across all experiments ------\n")

    plots.scatter_plots(params, all_measures) 
