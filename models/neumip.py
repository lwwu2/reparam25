import torch
import torch.nn as nn

from .mlps import make_mlp, PositionalEncoding
from .cumlp import mlp_fused32, mlp_serialization

""" Neural BRDF implementation"""


def fetch2D(texture,uv):
    """ fetch neumip textures
    Args:
        texutre: BxHxH
        uv: Bx2
    Return:
        BxT texture
    """
    B,_ = uv.shape
    _,H,_ = texture.shape
    
    uv[...,1] *= -1
    uv *= (H-1)
    
    uv0 = uv.floor().long()
    uv1 = uv.ceil().long()%H
    uv -= uv0
    uv0 %= H
    
    u0,v0 = uv0.T
    u1,v1 = uv1.T
    u,v = uv.T
    
    ret = (texture[:,v0,u0]*(1-u)*(1-v)+texture[:,v0,u1]*u*(1-v)\
        + texture[:,v1,u0]*(1-u)*v+texture[:,v1,u1]*u*v).T
    
    return ret



class NeuMIP(nn.Module):
    """ NeuMIP BRDF model"""
    def __init__(self,res=(512,512)):
        super(NeuMIP,self).__init__()
        self.offset_network = make_mlp(8+2,1,32,2,act='LeakyReLU')
        self.rgb_network = make_mlp(8+4,3,32,2,act='LeakyReLU')
        self.register_parameter('offset_texture',nn.Parameter(torch.randn(8,*res)*0.01))
        self.register_parameter('rgb_texture',nn.Parameter(torch.randn(8,*res)*0.01))

    def get_displace(self,uv,wo):
        """ displace the uv and return the rgb texture """
        f_offset = fetch2D(self.offset_texture,uv.clone())
        r = self.offset_network(torch.cat([f_offset,wo[...,:2]],-1))
        uv_offset = (r/(1-wo.pow(2).sum(-1,keepdim=True)).clamp_min(0.36).sqrt())*wo
        
        uv_new = (uv_offset + uv)
        f_rgb = fetch2D(self.rgb_texture,uv_new.clone())
        
        return f_rgb
    
    def get_rgb(self,f_rgb,wo,wi):
        """ return BRDF response """
        return self.rgb_network(torch.cat([wi,wo,f_rgb],-1)).relu()
    
    def forward(self,uv,wo,wi):
        """ query the neumip BRDF
        Args:
            uv: Bx2 in [0,1]
            wo: Bx2 in [-1,1]
            wi: Bx2 in [-1,1]
        """
        f_offset = fetch2D(self.offset_texture,uv.clone())
        r = self.offset_network(torch.cat([f_offset,wo[...,:2]],-1))
        uv_offset = (r/(1-wo.pow(2).sum(-1,keepdim=True)).clamp_min(0.36).sqrt())*wo
        uv_new = (uv_offset + uv)
        f_rgb = fetch2D(self.rgb_texture,uv_new.clone())
        btf = self.rgb_network(torch.cat([wi,wo,f_rgb],-1)).relu()
        return btf
    
    
    """ mitsuba api"""
    def prepare(self,):
        """ convert pytorch mlp weights """
        self.register_buffer('offset_params',mlp_serialization(self.offset_network).half())
        self.register_buffer('rgb_params',mlp_serialization(self.rgb_network).half())
        
        del self.offset_network # no longer used
        del self.rgb_network
        torch.cuda.empty_cache()
        return 

    def eval_texture(self,uv,wo):
        """ CUDA version of get_displace """
        f_offset = fetch2D(self.offset_texture,uv.clone())
        r = mlp_fused32(1,2,1,torch.cat([f_offset,wo[...,:2]],-1).half(),self.offset_params).float()
        uv_offset = (r/(1-wo.pow(2).sum(-1,keepdim=True)).clamp_min(0.36).sqrt())*wo
        uv_new = (uv_offset + uv)
        f_rgb = fetch2D(self.rgb_texture,uv_new.clone())
        return f_rgb
    
    def eval(self,f_rgb,wo,wi):
        """ CUDA vesion of get_rgb """
        return mlp_fused32(3,2,1,torch.cat([wi,wo,f_rgb],-1).half(),self.rgb_params).float().relu()




class NeuBTF(nn.Module):
    """ Neural BRDF without the spatial inputs"""
    def __init__(self,C=32,D=2,L=4,act='SiLU'):
        super(NeuBTF,self).__init__()
        self.encode = PositionalEncoding(L)
        self.mlp = make_mlp(2*(L*2+1) + 2,3,C,D,act=act)
    
    def get_displace(self,uv,wo):
        return torch.zeros(len(uv),0,device=uv.device)
    
    def get_rgb(self,f_rgb,wo,wi):
        cond = self.encode(wo)
        btf = self.mlp(torch.cat([cond,wi],-1))
        return btf.exp()

    def forward(self,uv,wo,wi):
        """ query the BRDF, uv is not used
        """
        cond = self.encode(wo)
        btf = self.mlp(torch.cat([cond,wi],-1))
        return btf.exp()
    

    """ mitsuba api"""
    def prepare(self,):
        """ convert pytorch mlp weights """
        self.register_buffer('params',mlp_serialization(self.mlp).half())
        
        del self.mlp # no longer used
        torch.cuda.empty_cache()
        return 
    
    def eval_texture(self,uv,wo):
        """ CUDA version of get_displace """
        return torch.zeros(len(uv),0,device=uv.device)
    
    def eval(self,uv,wo,wi):
        """ CUDA vesion of get_rgb """
        cond = self.encode(wo)
        inputs = torch.cat([cond,wi],-1).half()
        btf = mlp_fused32(3,2,2,inputs,self.params).float()
        return btf.exp()