"""
meta_atom_rnn/utils/network/train.py
Author: Andy
"""


import lightning as L
import os
import time
import yaml

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from utils.network.models import Network

from utils.data import load_data

from utils.general import create_folder

def run(params):

    if params['deployment_mode'] == 0:
        experiment_path = os.path.join(params['mounted_paths']['results']['checkpoints'], "k_" + str(params['dataset']['seq_len']).zfill(2))
    elif params['deployment_mode'] == 1:
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

    arch_val = params['network']['arch']
    arch_str = "rnn/" if arch_val == 0 else 'lstm/'

    print(f"path_save = {path_save}")
    exp_logger = CSVLogger(save_dir=path_save,version=arch_str)

    # Create: Trainer

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger,)
                        #default_root_dir=folder_name)

    # Train: Model

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)

    #yaml.dump(params, open(os.path.join(params_path), 'w'))

