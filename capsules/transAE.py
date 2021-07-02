"""
This file defines transforming auto-encoder architecture

Created by Kunhong Yu
Date: 2021/07/01
"""
import torch as t
from capsules.capsule import Capsule


class transAE(t.nn.Module):
    """Define TransAE architecture"""

    def __init__(self, input_dim : int, cap_dim : int, out_dim : int, num_caps : int):
        """
        Args :
            --input_dim: input dimension
            --cap_dim: capsule hidden dimension
            --out_dim: output dimension
            --num_caps: number of capsules
        """
        super(transAE, self).__init__()

        self.input_dim = input_dim
        self.cap_dim = cap_dim
        self.out_dim = out_dim
        self.num_caps = num_caps

        layers = [Capsule(self.input_dim, self.cap_dim, self.out_dim) for _ in range(self.num_caps)]

        self.transae_layer = t.nn.Sequential(*layers)

        self.sigmoid = t.nn.Sigmoid()

    def forward(self, x, xy):
        """x has dimension (-1, 784) for MNIST, xy is tuple, (delta_x, delta_y)"""
        res = []
        for layer in self.transae_layer:
            x_re = layer(x, xy)
            res.append(x_re)

        for i, x_re in enumerate(res):
            if i == 0:
                whole = x_re

            else:
                whole = whole + x_re

        out = self.sigmoid(whole)

        return out