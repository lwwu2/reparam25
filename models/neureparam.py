import torch
import torch.nn as nn
import torch.nn.functional as NF

import math

from .mlps import make_mlp, PositionalEncoding
from .cumlp import mlp_fused16, mlp_reparam16, mlp_serialization, mlp_grad_serialization


def box_muller(sample2):
    """ box muller transformation that maps a 2D uniform sample to 2D gaussians"""
    r = (-2*torch.log(sample2[...,0].clamp_min(1e-12))).sqrt()
    theta = 2*math.pi*sample2[...,1]
    return torch.stack([
        torch.cos(theta)*r,
        torch.sin(theta)*r
    ],-1)

def grad(y,x):
    d_output = torch.ones_like(y,requires_grad=False)
    y_grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
    only_inputs=True)[0]
    return y_grad


class NeuReparam(nn.Module):
    """ Implementation of the reparameterization model"""
    def __init__(self, C1, D1, C2, D2, L=4, T=0):
        """
        Args:
            C1/C2: hidden dim of the reparameterization/pdf network
            D1/D2: hidden layers of the reparameterization/pdf network
            L: positional encoding frequency
            T: neural texture feature (for spatially varying BRDF)
        """
        super(NeuReparam,self).__init__()
        
        self.encode = PositionalEncoding(L)
        
        Cond = (L*2+1)*2+T
        self.mlp = make_mlp(Cond+2,3,C1,D1,act='SiLU')
        self.mlp2 = make_mlp(Cond+2,1,C2,D2,act='ReLU')
        self.T = T



    def encode_cond(self,wo,f):
        """ encode conditions (omega_o, neural texture)
        Args:
            wo: Bx2 viewing direction
            f: BxT neural texture
        Return:
            Bx((2*L+1)*2+T) condition feature
        """
        #if f is None:
        #    return self.encode(wo)
        #else:
        return torch.cat([f,self.encode(wo)],-1)


    def forward(self,cond,z):
        """ compute reparameterization given the prior sample and condition feature
        Return:
            wi: Bx2
        """
        Tz = self.mlp(torch.cat([cond,z],-1))
        
        # map to hemisphere
        wi = torch.cat([Tz[...,:2],NF.softplus(Tz[...,2:])],-1)
        wi = NF.normalize(wi,dim=-1)
        return wi[...,:2]
    
    def detJ(self,cond,z):
        """ compute the reparameterization Jacobian determinant
        Args:
            z: Bx2 prior samples
            cond: condition feature
        Return:
            wi: Bx2 reparameterized samples
            Jac: B corresponding Jacobian determinant
        """
        z_grad = z.clone()
        z_grad.requires_grad=True

        wi = self.forward(cond,z_grad)
        detJ = torch.stack([grad(wi[...,0],z_grad),grad(wi[...,1],z_grad)],-1).det()
        return wi,detJ
    
    def pdf(self,cond,wi):
        """ compute approximated pdf
        """
        pdf = self.mlp2(torch.cat([cond,wi],-1))
        if self.T == 0:
            pdf = pdf.exp()
        return pdf.squeeze(-1)
    

    """ mitsuba api """
    def prepare(self,):
        """ convert pytorch mlp weights """
        self.register_buffer('params',mlp_grad_serialization(self.mlp).half())
        self.register_buffer('params2',mlp_serialization(self.mlp2).half())

        del self.mlp
        del self.mlp2
        torch.cuda.empty_cache()
        return
    
    def pdf_cond(self,cond,wi):
        """ pdf evaluation in CUDA """
        pdf = mlp_fused16(1,1,0,torch.cat([cond,wi[...,:2]],-1).half(),
                          self.params2)
        if self.T == 0:
            pdf = pdf.exp()
        pdf = pdf.relu().squeeze(-1).float()
        return pdf*wi[...,-1].relu()
    
    def sample_cond(self,cond,sample2):
        """ pdf sampling in CUDA """
        z = box_muller(sample2)

        wi,detJ = mlp_reparam16(torch.cat([cond,z],-1).half(),self.params)
        wi,detJ = wi.float(),detJ.float()
        
        weight = detJ*(2*math.pi)*torch.exp(0.5*z.pow(2).sum(-1))/wi[...,-1].relu()
        
        pdf = mlp_fused16(1,1,0,torch.cat([cond,wi[...,:2]],-1).half(),
                          self.params2)
        if self.T == 0:
            pdf = pdf.exp()
        pdf = pdf.relu().squeeze(-1).float()
        
        return wi, pdf*wi[...,-1].relu(), weight




class Diffuse(nn.Module):
    """ Cosine-weighted BRDF sampling"""
    def __init__(self,):
        super(Diffuse,self).__init__()
        return
    
    def prepare(self,):
        return None

    def encode_cond(self,wo,f_rgb):
        return 0
    
    def sample_cond(self,cond,sample2):
        r2 = sample2[...,0]
        cos_theta = (1-r2).sqrt()
        r = r2.sqrt()
        theta = sample2[...,1]*2*math.pi
        
        wi = torch.stack([
            r*torch.cos(theta),
            r*torch.sin(theta),
            cos_theta
        ],-1)
        
        pdf = cos_theta/math.pi
        weight = 1/pdf
        return wi,pdf,weight
    
    def pdf_cond(self,cond,wi):
        return wi[...,-1].relu()/math.pi