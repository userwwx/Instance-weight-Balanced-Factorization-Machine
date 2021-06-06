import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, normal_


class AttLayer(nn.Module):
    def __init__(self, in_dim, att_dim, temperature=1.0):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.temperature = temperature
        self.w = nn.Linear(in_dim, att_dim)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)
        att_signal = F.relu(att_signal)
        att_signal = torch.mul(att_signal, self.h)
        att_signal = torch.sum(att_signal, dim=2)
        att_signal = F.softmax(att_signal / self.temperature, dim=1)

        return att_signal


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0, activation='relu', bn=False, init_mothod=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_mothod = init_mothod

        mlp_modules = []
        for idx, (intput_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(intput_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_mothod is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_mothod == 'norm':
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(module.weight.data, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name='relu', emb_dim=None):
    """Construct activation layers
    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation
    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation