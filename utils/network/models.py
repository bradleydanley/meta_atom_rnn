"""
Purpose: Temporal Models
Author: Andy
"""


import torch
import lightning as L

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.network.convlstm import ConvLSTM
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomCheckpointCallback(ModelCheckpoint):
    
    def __init__(self, folder_name, arch):

        super().__init__()
        self.folder_name = folder_name

    def _get_metric_interpolated_filepath_name(self, interpolation_values, epoch, logs):

        return f'{self.folder_name}_{epoch:02d}_{arch}'

class Network(L.LightningModule):

    def __init__(self, params):

        super().__init__()

        self.choice = params["network"]["arch"]
        self.alpha = params["network"]["learning_rate"]
        self.num_epochs = params["network"]["num_epochs"]
        self.batch_size = params["network"]["batch_size"]

        self.create_architecture(params)

    def create_architecture(self, params):

        seq_len = params["dataset"]["seq_len"]

        if self.choice == 0 or self.choice == 1:

            self.i_dims = params["rnn_lstm"]["i_dims"]
            self.h_dims = params["rnn_lstm"]["h_dims"]
            self.num_layers = params["rnn_lstm"]["num_layers"]

            if self.choice == 0:
                self.name = "rnn"
                self.arch = torch.nn.RNN(self.i_dims, self.h_dims,
                                         self.num_layers, batch_first=True)
            else:
                self.name = "lstm"

                # self.num_layers makes us stack the layers
                self.arch = torch.nn.LSTM(self.i_dims, self.h_dims,
                                          self.num_layers, batch_first=True)

            # tanh() b/c [-1, -1] fits our problem
            self.linear = torch.nn.Sequential(torch.nn.Linear(self.h_dims,
                                                              self.i_dims),
                                              torch.nn.Tanh())

        elif self.choice == 2:

            spatial = params["convlstm"]["spatial"]
            padding = params["convlstm"]["padding"]
            k_size = params["convlstm"]["kernel_size"]
            in_channels = params["convlstm"]["in_channels"]
            out_channels = params["convlstm"]["out_channels"]
            num_layers = params["convlstm"]["num_layers"]

            spatial = (spatial, spatial)
            self.name = "convlstm"

            self.arch = torch.nn.Sequential()

            for i in range(num_layers):

                name = "convlstm_%s" % i
                layer = ConvLSTM(out_channels=out_channels,
                                 in_channels=in_channels,
                                 padding=padding, seq_len=seq_len,
                                 kernel_size=k_size, frame_size=spatial)

                self.arch.add_module(name, layer)

            self.arch.add_module("tanh", torch.nn.Tanh())

    def configure_optimizers(self):

        # Create: Optimzation Routine

        optimizer = Adam(self.parameters(), lr=self.alpha)

        # Create: Learning Rate Schedular

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def objective(self, preds, labels):

        criterion = torch.nn.MSELoss()

        return criterion(preds, labels)

    def training_step(self, batch, batch_idx):

        samples, labels = batch

        # Gather: Predictions

        # rnn is going to make [target_seq] predictions. how much will we prop forward through time?
        # this is the 'many' part of the second many in many to many LOL
        # first part of many to many is in utils/data.py
        target_seq = labels.size()[1]

        preds = self(samples, target_seq)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("train_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        samples, labels = batch

        # Gather: Predictions

        target_seq = labels.size()[1]

        preds = self(samples, target_seq)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("valid_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

    def forward(self, x, target_seq):

        batch, seq_len, channel, height, width = x.size()

        all_pred = torch.zeros(batch, target_seq, channel,
                               height, width).to(x.device)

        # Forward: RNN, LSTM

        if self.name == "rnn" or self.name == "lstm":

            x = x.view(batch, seq_len, -1)
            h = torch.zeros(self.num_layers, batch,
                            self.h_dims).to(x.device)

            if self.name == "lstm":
                c = torch.zeros(self.num_layers, batch,
                                self.h_dims).to(x.device)
                meta = (h, c)
            else:
                meta = h

            for i in range(target_seq):
                pred, meta = self.arch(x, meta)
                pred = self.linear(pred.reshape(batch, -1))
                pred = pred.view((batch, channel, height, width))
                all_pred[:, i, :, :, :] = pred

            # pred, meta = self.arch(x, meta)
            # pred = self.linear(pred)
            # all_pred = pred.view((batch, target_seq, channel, height, width))

        # Forward: ConvLSTM

        else:

            all_pred = self.arch(x)

        return all_pred

