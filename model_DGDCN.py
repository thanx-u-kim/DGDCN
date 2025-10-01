import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCNLayer(Module):
    def __init__(self, in_dim: int, conv_dim: int):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.conv_dim = conv_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, conv_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
                
    def forward(self, inputs, adj):
        batch_size = inputs.shape[0]
        num_nodes = adj.shape[1]
        inputs = inputs.transpose(1, 2)  # [B, 1, N] -> [B, N, 1]
        AX = torch.bmm(adj, inputs)      # [B, N, N] @ [B, N, 1] -> [B, N, 1]
        AX = AX.reshape((batch_size * num_nodes, self.in_dim))  # [B*N, 1]
        AXW = AX @ self.weight           # [B*N, 1] @ [1, C] -> [B*N, C]
        AXW = AXW.reshape((batch_size, num_nodes, self.conv_dim))  # [B, N, C]
        output = AXW.transpose(1, 2)     # [B, C, N]
        return output
    
    def __repr__(self):
        return f"{self.__class__.__name__} (in={self.in_dim} -> out={self.conv_dim})"


class DGDCN(pl.LightningModule):
    def __init__(self, metricnormalization, dijkstra, conv_channels, num_nodes, pred_len, out_channels, dropout, lr, weight_decay):
        '''
        : param conv_channels: the number of convolution weights
        : param gru_channels: the number of hiddens state of GRU
        : param out_channels: the number of nodes
        '''
        super().__init__()
        self.metricnormalization = metricnormalization
        self.dijkstra = dijkstra
        self.conv_channels = conv_channels
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.in_GC1 = GCNLayer(1, self.conv_channels)
        self.out_GC1 = GCNLayer(1, self.conv_channels)
        self.SpatialConvolution = nn.Conv2d(1, self.pred_len, (2*self.conv_channels,1)) ## 1ch -> 6ch
        self.GRU = nn.GRU(self.num_nodes, self.out_channels, num_layers=1, batch_first=True)
        self.regressor = nn.Linear(self.out_channels, self.out_channels, bias=True)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, in_adj, out_adj):
        batch_size, train_len, num_nodes = x.shape[0], x.shape[1], x.shape[2]
        
        conv_output = None
        gru_output = None

        for i in range(train_len):
            _x_input = x[:, i, :].reshape(batch_size, 1, num_nodes)

            _in_adj = in_adj * self.dijkstra
            _in_adj = F.normalize(_in_adj, p=1, dim=2, eps=0)
            _inflow_x = F.dropout(_x_input, self.dropout, training=self.training)
            _inflow_x = F.relu(self.in_GC1(_inflow_x, _in_adj))  # [B, C, N]

            _out_adj = out_adj * self.dijkstra
            _out_adj = F.normalize(_out_adj, p=1, dim=2, eps=0)
            _outflow_x = F.dropout(_x_input, self.dropout, training=self.training)
            _outflow_x = F.relu(self.out_GC1(_outflow_x, _out_adj))  # [B, C, N]

            _conv = torch.cat([_inflow_x, _outflow_x], dim=1)        # [B, 2C, N]
            _conv = _conv.reshape((batch_size, 1, -1, num_nodes))    # [B, 1, 2C, N]
            _conv = self.SpatialConvolution(_conv)                   # [B, pred_len, 1, N]

            conv_output = _conv if conv_output is None else torch.cat([conv_output, _conv], dim=2)

        for j in range(self.pred_len):
            temporal_part = conv_output[:,j,:,:]
            _, hidden = self.GRU(temporal_part)
            output = hidden[-1]
            output = output.reshape((batch_size, -1, num_nodes)) ## [batch_size, 1, num_nodes]
            if gru_output is None:
                gru_output = output
            else:
                gru_output = torch.cat([gru_output, output], dim=1) ## [batch_size, pred_len, num_nodes]
        tot_output = self.regressor(gru_output)
        return tot_output

    def training_step(self, batch, batch_idx):
        x, y, in_a, out_a = batch
        output = self(x, in_a, out_a)
        loss = F.mse_loss(output, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, in_a, out_a = batch
        output = self(x, in_a, out_a)
        loss = F.mse_loss(output, y)
        RMSE = torch.sqrt(loss)
        MAE = F.l1_loss(output, y)
        metrics = {
            'val_loss': loss,
            'val_RMSE': RMSE * self.metricnormalization,
            'val_MAE': MAE * self.metricnormalization
        }
        self.log_dict(metrics)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, in_a, out_a = batch
        output = self(x, in_a, out_a)
        loss = F.mse_loss(output, y)
        RMSE = torch.sqrt(loss)
        MAE = F.l1_loss(output, y)
        metrics = {
            'test_RMSE': RMSE * self.metricnormalization,
            'test_MAE': MAE * self.metricnormalization
        }
        self.log_dict(metrics)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, in_a, out_a = batch
        predictions = self(x, in_a, out_a)
        return predictions
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer