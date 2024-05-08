"""
Purpose: Train Deep Learning System
Author: Andy
"""


import lightning as L
import os
import time

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.data import load_data
#from utils.network.models import Network, CustomCheckpointCallback 
from utils.network.models import Network, ModelCheckpoint 

from utils.general import create_folder

def run(params):

    #experiment_path = os.path.join(params['paths']['results'], "k_" + str(params['dataset']['seq_len']))
    experiment_path = os.path.join(params['kube']['train_job']['paths']['results']['model_checkpoints'], "k_" + str(params['dataset']['seq_len']).zfill(2))

    create_folder(experiment_path) 
    path_save = os.path.join(experiment_path)
    num_epochs = params["network"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]

    # Load: Datasets

    start_time = time.time()
    train, valid = load_data(params)
    load_time = time.time() - start_time
    # Create: Model
    
    model = Network(params)

    # Create: Logger

    print(f"path_save = {path_save}")
    exp_logger = CSVLogger(save_dir=path_save)

    # Create: Trainer

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    arch = f"./{params['network']['arch']}"
    folder_name = "rnn/" if arch == 0 else 'lstm/'
    #checkpoint_callback = CustomCheckpointCallback(folder_name=folder_name, arch=folder_name)
    checkpoint_callback = ModelCheckpoint(dirpath=folder_name)

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger,
                        default_root_dir=folder_name)

    # Train: Model

    start_time = time.time()
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)
    train_time = start_time - time.time() 

    return load_time, train_time
