import torch
import math


def binning2d(x,y,res,x0,x1,y0,y1,drop=True):
    """ binning samples into 2D histogram
    Args:
        x,y: sample x,y coordinate
        res: histogram resolution
        x0,x1,y0,y1: bound of the sample
        drop: drop sample if it is outside the bound
    """
    if drop:
        valid = (x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
        x = x[valid]
        y = y[valid]
    x = ((x-x0)/(x1-x0)*res).long().clamp(0,res-1)
    y = ((y-y0)/(y1-y0)*res).long().clamp(0,res-1)
    idx = x + y*res
    hist = torch.zeros(res*res,device=idx.device)
    hist.scatter_add_(0,idx,torch.ones_like(idx).float())
    hist /= hist.sum()
    hist = hist.reshape(res,res)
    return hist


def angle2xyz(theta,phi,degree=True):
    """" spherical coordinates to solid angle"""
    if degree:
        theta = theta/180*math.pi
        phi = phi/180*math.pi
    z = torch.cos(theta)
    r = torch.sin(theta)
    return torch.stack([
        r*torch.cos(phi),
        r*torch.sin(phi),
        z
    ],-1)