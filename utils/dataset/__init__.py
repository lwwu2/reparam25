import torch
from torch.utils.data import Dataset
import math


class PositionDirectionDataset(Dataset):
    """ Training dataset used to stratified sample input conditions (u,omega_o)"""
    def __init__(self,batch_size,spatial,angular,device='cpu'):
        """
        Args:
            batch_size: batch_size of each training step
            spatial: spatial stratifications
            angular: angular stratifications
            device: GPU index
        """
        self.S = spatial
        self.A = angular
        self.length = self.S*self.S*self.A*self.A
        assert self.length % batch_size == 0
        self.length //= batch_size
        self.batch_size = batch_size
        self.device=device
        self.inds = torch.arange(self.length*batch_size,device=device) # indices for the strata
        
    def __len__(self,):
        return self.length
    
    def resample(self,):
        """ shuffle the strata indices """
        self.inds = torch.randperm(self.length*self.batch_size,device=self.device)
    
    def __getitem__(self,idx):
        """
        Return:
            wo: Bx2 viewing direction sample
            u: Bx2 uv coordinates
        """

        # obtain the strata index
        idx = self.inds[idx*self.batch_size:(idx+1)*self.batch_size]
        aij = idx % (self.A*self.A)
        sij = idx // (self.A*self.A)
        
        
        # uniformly sampling the uv coordinates
        u = torch.stack([sij//self.S,sij%self.S],-1).float()\
          + torch.rand(self.batch_size,2,device=self.device)
        u /= self.S
        
        # cosine-weighted sampling omega_o
        wo = torch.stack([aij//self.A,aij%self.A],-1).float()\
           + torch.rand(self.batch_size,2,device=self.device)
        wo /= self.A
        r,theta = wo[...,0].sqrt(),wo[...,1]*2*math.pi
        wo = torch.stack([
            r*torch.cos(theta),
            r*torch.sin(theta)
        ],-1)
        
        return {
            'wo': wo,
            'u': u
        }
