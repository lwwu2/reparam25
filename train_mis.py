
import torch
import torch.nn.functional as NF
from torch.utils.data import DataLoader
import torchvision

import lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


import math

from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from models.neumip import NeuMIP, NeuBTF
from models.neureparam import NeuReparam, box_muller
from utils import binning2d
from utils.dataset import PositionDirectionDataset


class ModelTrainer(pl.LightningModule):
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer,self).__init__()
        self.save_hyperparameters(hparams)
        
        # load training ckpt
        state_dict = torch.load(hparams.model_ckpt,map_location='cpu')['state_dict']
        
        # load neural BRDF
        weight = {}
        for k,v in state_dict.items():
            if 'neumip.' in k:
                weight[k.replace('neumip.','')] = v
        if 'rgb_texture' in weight.keys():
            self.neumip = NeuMIP()
            T = 8
        else:
            self.neumip = NeuBTF()
            T = 0
        self.neumip.load_state_dict(weight)
        self.neumip.requires_grad_(False)

        # load reparameterization model
        weight = {}
        for k,v in state_dict.items():
            if 'model.' in k:
                weight[k.replace('model.','')] = v
        self.model = NeuReparam(**hparams.model,T=T)
        self.model.load_state_dict(weight)
        self.model.mlp.requires_grad_(False)
        

        self.train_dataset = PositionDirectionDataset(hparams.batch_size,hparams.spatial,hparams.angular,device=hparams.device)
        self.val_dataset = PositionDirectionDataset(1,4,1,device=hparams.device) # 16 BRDF lobe images
        # how many samples per sampled condition
        spp_ = int(hparams.spp**0.5)
        self.spp = spp_*spp_
        self.spp_ = spp_
        self.strata = torch.stack(torch.meshgrid(*[torch.arange(spp_,device=hparams.device)]*2,indexing='xy'),-1).reshape(1,-1,2).float()
        
        # validation image buffer
        self.val_buffer = torch.zeros(self.val_dataset.length,64*2,64)
        
    def __repr__(self,):
        return repr(self.hparams)
    
    def configure_optimizers(self,):
        opt = torch.optim.Adam
        
        optimizer = opt(self.model.parameters(),lr=self.hparams.learning_rate_mis)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=self.hparams.milestones, gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        return DataLoader(self.train_dataset, shuffle=False, batch_size=None)
    
    def val_dataloader(self,):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=None)
    
    def forward(self,x):
        return
    
    def on_train_epoch_start(self,):
        # resample stratas every epoch
        self.train_dataset.resample()
        torch.cuda.empty_cache()
        return
        
    
    def training_step(self, batch, batch_idx):
        u,wo = batch['u'],batch['wo']
        
        B = u.shape[0]
        

        # neural feature and input condition
        f_rgb = self.neumip.get_displace(u,wo)
        cond = self.model.encode_cond(wo,f_rgb).repeat_interleave(self.spp,dim=0)
        f_rgb = f_rgb.repeat_interleave(self.spp,dim=0)
        wo = wo.repeat_interleave(self.spp,dim=0)

        # sample prior distribution
        z = box_muller((self.strata+torch.rand(B,self.spp,2,device=wo.device))
                        /self.spp_).reshape(-1,2)
        
        # compute the reparameterization
        wi,detJ = self.model.detJ(cond,z)
        wi,detJ = wi.detach(),detJ.detach().abs()

        # compute the pdf approximation
        pdf = self.model.pdf(cond,wi)*detJ
        pdf_gt = torch.exp(-0.5*z.pow(2).sum(-1)).div(2*math.pi)


        if self.hparams.log_loss:
            loss = NF.l1_loss(torch.log(1e-14+pdf_gt),
                                      torch.log(1e-14+pdf))
        else:
            loss = NF.l1_loss(pdf_gt,pdf)
            #loss = NF.mse_loss(pdf_gt,pdf)
        
        if loss.isnan() or loss.isinf():
            print("Get NAN!")
            return None
        
        self.log('train/loss',loss, prog_bar=True)

        return loss
    
        
    def validation_step(self, batch, batch_idx):
        u,wo = batch['u'],batch['wo']
        res = 64

        # gt
        wi = torch.meshgrid(*[torch.linspace(-1,1,res,device=u.device)]*2,indexing='xy')
        wi = torch.stack(wi,-1).reshape(-1,2)
        
        # pdf approximation
        f_rgb = self.neumip.get_displace(u,wo)
        cond = self.model.encode_cond(wo,f_rgb)
        pdf = self.model.pdf(cond.expand(len(wi),-1),wi).relu()
        pdf[wi.pow(2).sum(-1)>1] = 0
        pdf = pdf.reshape(res,res)
        pdf /= pdf.max()
        
        # gt by binning
        pdf_gt = torch.zeros(res,res,device=wo.device)
        B=8
        for _ in range(B):
            z = torch.randn(1000_000,2,device=wo.device)
            wi = self.model(cond.expand(len(z),-1),z)
            pdf_gt += binning2d(wi[...,0],wi[...,1],res,-1,1,-1,1)
        pdf_gt /= B
        pdf_gt /= pdf_gt.max()
        
        loss = NF.mse_loss(pdf,pdf_gt)
        self.val_buffer[batch_idx] = torch.cat([pdf,pdf_gt],0).cpu()
        self.log('val/loss',loss)
        return 
    
    def on_validation_epoch_end(self,):
        images = self.val_buffer.reshape(2,8,128,64).permute(0,2,1,3).reshape(2*128,8*64)
        torchvision.utils.save_image(images.pow(1/2.2),'{}/{}/ep{:04d}.png'.format(
                self.hparams.log_path,self.hparams.experiment_name,self.current_epoch))
        
    
if __name__ == '__main__':
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/rgl.yaml')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.set_defaults(resume=False)
    
    # merge command line argument with yaml argument
    args,_ = parser.parse_known_args()
    args = OmegaConf.load(args.config)
    for k,v in args.items():
        parser.add_argument('--{}'.format(k),type=type(v),default=v)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    experiment_name = args.experiment_name
    log_path = Path(args.log_path)
    logger = TensorBoardLogger(log_path, name=experiment_name)
    
    checkpoint_path = log_path / experiment_name
    checkpoint_path.mkdir(parents=True,exist_ok=True)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=4, save_last=True, every_n_epochs=1)
    
    
    
    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    
    # setup model trainer
    model = ModelTrainer(args)
    
    trainer = Trainer(
        accelerator='gpu', devices=[args.device], 
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1
    )
    
    trainer.fit(
        model, 
        ckpt_path=last_ckpt, 
    )