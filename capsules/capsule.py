"""
This file builds capsule architecture in the paper
Code inspiration: https://github.com/IsCoelacanth/TransformingAutoencoder_PyTorch/blob/master/Capsule.py

Created by Kunhong Yu
Date: 2021/07/01
"""
import torch as t


class Capsule(t.nn.Module):
    """define capsule module"""

    def __init__(self, input_dim : int, cap_dim : int, out_dim : int):
        """
        Args :
            --input_dim: input dimension
            --cap_dim: capsule hidden dimension
            --out_dim: output dimension
        """
        super(Capsule, self).__init__()

        self.input_dim = input_dim
        self.cap_dim = cap_dim
        self.out_dim = out_dim

        self.reco_layer = t.nn.Sequential(t.nn.Linear(self.input_dim, self.cap_dim), t.nn.Sigmoid()) # recognition layer
        self.xy_layer = t.nn.Linear(self.cap_dim, 2) # xy layer
        self.prob_layer = t.nn.Sequential(t.nn.Linear(self.cap_dim, 1), t.nn.Sigmoid()) # probability layer
        self.gen_layer = t.nn.Sequential(t.nn.Linear(2, self.out_dim), t.nn.ReLU(inplace = True)) # generation layer
        self.output_layer = t.nn.Sequential(t.nn.Linear(self.out_dim, self.input_dim)) # reconstruction layer

    def forward(self, x, xy_shift):
        """x has dimension (-1, 784) for MNIST, xy is tuple, (delta_x, delta_y)"""
        reco = self.reco_layer(x)
        xy = self.xy_layer(reco)
        prob = self.prob_layer(reco)

        xy = xy + xy_shift

        gen = self.gen_layer(xy)
        out = self.output_layer(gen) # [m, 784]
        out = prob * out # broadcasting

        return out