import torch
import torch.nn as nn
import math


def make_mlp(Ci,Co,C,D,act='ReLU'):
    """
    Args:
        Ci: input dim
        Co: output dim
        C: hidden dim
        D: hidden layers
        act: activation function
    """
    act = getattr(nn,act)
    mlps = [nn.Linear(Ci,C),act()]
    for _ in range(D):
        mlps.append(nn.Linear(C,C))
        mlps.append(act())
    mlps.append(nn.Linear(C,Co))
    return nn.Sequential(*mlps)


class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ Positional encoding inputs to sines and cosines
        Args:
            L: number of frequency bands
        """
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        freq = 2**torch.arange(self.L,device=inputs.device)*math.pi
        enc = inputs[:,None]*freq[None,:,None]
        ret = torch.cat([
            torch.sin(enc),
            torch.cos(enc)
        ],-1).flatten(1)
        return torch.cat([inputs,ret],-1)