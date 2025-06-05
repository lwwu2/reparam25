import torch
import torch.nn as nn
import torch.nn.functional as NF
from torch.utils.cpp_extension import load
from pathlib import Path

""" cuda mlp inference implementation """


_ext_src_root = Path(__file__).parent / 'src'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = [
    "-O3",
    '-U__CUDA_NO_HALF_OPERATORS__', 
    '-U__CUDA_NO_HALF_CONVERSIONS__', 
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

_ext = load(name='cumlp_ext', 
            sources=_ext_src_files,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths)

__all__ = ["_ext"]


def mlp_fused16(Co,D,act_mode,inputs,params):
    """ mlp with hidden dim = 16
    Args:
        Co: output dim
        D: hidden layers
        act_mode: activation function 0: ReLU, 1: LeakyReLU, 2: SiLU
        inputs: BxCi input tensor
        params: serialized network parameters
    """
    B,Ci = inputs.shape
    Co_ = (-Co%16)
    Ci_ = -Ci%16
    B_ = -B%128
    inputs = torch.nn.functional.pad(inputs,(0,Ci_,0,B_))
    outputs = torch.zeros(B+B_,Co+Co_,device=inputs.device,dtype=inputs.dtype)
    _ext.mlp_fused16(D,act_mode,inputs.contiguous(),outputs.contiguous(),params.contiguous())
    return outputs[:B,:Co]

def mlp_fused32(Co,D,act_mode,inputs,params):
    """ mlp with hidden dim = 32
    """
    B,Ci = inputs.shape
    Co_ = (-Co%16)
    Ci_ = -Ci%16
    B_ = -B%128
    inputs = torch.nn.functional.pad(inputs,(0,Ci_,0,B_))
    outputs = torch.zeros(B+B_,Co+Co_,device=inputs.device,dtype=inputs.dtype)
    _ext.mlp_fused32(D,act_mode,inputs.contiguous(),outputs.contiguous(),params.contiguous())
    return outputs[:B,:Co]

def mlp_reparam16(inputs,params):
    """ mlp with hidden dim = 16 + forward Jacobian computation
    """
    B,Ci = inputs.shape
    Ci_ = -Ci%16
    B_ = -B%128
    inputs = torch.nn.functional.pad(inputs,(0,Ci_,0,B_))
    outputs = torch.zeros(B+B_,4,device=inputs.device,dtype=inputs.dtype)
    _ext.mlp_reparam16(inputs,outputs,params)
    outputs,detJ = outputs[:B,:3],outputs[:B,3]
    return outputs,detJ



def mlp_serialization(mlp):
    """ serialize mlp """
    ret = []
    for layer in mlp.children():
        if type(layer) is nn.Linear:
            h,w = layer.weight.shape
            h_ = -h%16
            w_ = -w%16
            
            # pad to 16
            bias = NF.pad(layer.bias.data,(0,h_))
            weight = NF.pad(layer.weight.data,(0,w_,0,h_))
            ret.append(bias)
            ret.append(weight.reshape(-1))
    ret = torch.cat(ret,0)
    return ret

def mlp_grad_serialization(mlp):
    """ serialize mlp with fwd gradient computation"""
    ret = []
    for i,layer in enumerate(mlp.children()):
        if type(layer) is nn.Linear:
            h,w = layer.weight.shape
            if i == 0:
                h_ = -h%16
                w_ = -w%16
                # first derivative corresponds to the model weight
                dx = layer.weight.data[:,-2] 
                dy = layer.weight.data[:,-1]
                ret.append(dx)
                ret.append(dy)

                # pad to 16
                bias = NF.pad(layer.bias.data,(0,h_))
                weight = NF.pad(layer.weight.data,(0,w_,0,h_))
            else:
                h_ = -h%8
                w_ = -w%8

                #pad to 8 as last layer has only 2-3 outputs
                bias = NF.pad(layer.bias.data,(0,h_))
                weight = NF.pad(layer.weight.data,(0,w_,0,h_))
            ret.append(bias)
            ret.append(weight.reshape(-1))
    ret = torch.cat(ret,0)
    return ret