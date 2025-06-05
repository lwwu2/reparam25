import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr


from .neureparam import NeuReparam, Diffuse
from .neumip import NeuMIP, NeuBTF


THRESH = 0.98

""" Neural BRDF implementation """
class NeuBSDF(mi.BSDF):
    def __init__(self,props):
        """
        props:
            uv_scale: texture scale
            device: GPU index
            mode: reparam or diffuse
            brdf_ckpt: neural BRDF checkpoint
            sampler_ckpt: reparameterization model checkpoint 
            C1/C2/D1/D2/L: reparameterization network config
        """
        mi.BSDF.__init__(self,props)
        self.uv_scale = props.get('uv_scale',1)

        mode = props['mode']
        device = torch.device(props.get('device',0))
        
        weight = torch.load(props['brdf_ckpt'],map_location='cpu')

        model_config = {
                    'C1': props.get('C1',16), 'D1': props.get('D1',2), 
                    'C2': props.get('C2',16), 'D2': props.get('D2',1),
                    'L': props.get('L',4)
        }

        # load neumip
        if 'rgb_texture' in weight.keys():
            neumip = NeuMIP()
            model_config['T'] = 8
            self.flip_sh_frame = True
        else:
            neumip = NeuBTF()
            model_config['T'] = 0
            self.flip_sh_frame = False
        neumip.load_state_dict(weight)
        neumip.to(device)
        neumip.requires_grad_(False)
        neumip.prepare()
        del weight
        
        
        # load reparameterization model
        if mode == 'diffuse':
            neusampler = Diffuse()
        else:
            ckpt_path = props['sampler_ckpt']
            # check post fix
            if ckpt_path.endswith('.ckpt'):
                weight = {}
                state_dict = torch.load(ckpt_path,map_location='cpu')['state_dict']
                for k,v in state_dict.items():
                    if 'model.' in k:
                        weight[k.replace('model.','')] = v
                del state_dict
            else:
                weight = torch.load(ckpt_path,map_location='cpu')
                
            neusampler = NeuReparam(**model_config)
            neusampler.load_state_dict(weight)
            del weight
        neusampler.to(device)
        neusampler.requires_grad_(False)
        neusampler.prepare()
        
        self.neusampler = neusampler
        self.neumip = neumip
        
        reflection_flags   = mi.BSDFFlags.SpatiallyVarying|mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components  = [reflection_flags]
        self.m_flags = reflection_flags
        
    def sample(self, ctx, si, sample1, sample2, active):
        flip = mi.Vector3f(1.0)
        # neumip material has flipped shading frame
        if self.flip_sh_frame:
            flip.x = dr.sign(dr.dot(si.sh_frame.s,si.dp_du))
            flip.y = dr.sign(dr.dot(si.sh_frame.t,si.dp_dv))
        
        uv_ = (si.uv.torch()*self.uv_scale)
        wi_ = (si.wi*flip).torch()[...,:2]
        
        f_rgb_ = self.neumip.eval_texture(uv_,wi_)
        cond_ = self.neusampler.encode_cond(wi_,f_rgb_)
        
        wo_,pdf_,weight_ = self.neusampler.sample_cond(cond_,sample2.torch())
        value_ = self.neumip.eval(f_rgb_,wi_,wo_[...,:2]).relu()*weight_.unsqueeze(-1)
        value_ *= (wo_[...,:2].pow(2).sum(-1,keepdim=True)<=THRESH)
        value_ = torch.where(value_.isnan(),0,value_)
        
        bs = mi.BSDFSample3f()
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.eta = 1.0
        bs.pdf = dr.select(active,mi.Float(pdf_),0)
        bs.wo = mi.Vector3f(wo_)*flip
        value = mi.Spectrum(value_)
        
        return (bs,value)
    
    def eval(self, ctx, si, wo, active):
        flip = mi.Vector3f(1.0)
        # neumip material has flipped shading frame
        if self.flip_sh_frame:
            flip.x = dr.sign(dr.dot(si.sh_frame.s,si.dp_du))
            flip.y = dr.sign(dr.dot(si.sh_frame.t,si.dp_dv))
        
        uv_ = (si.uv.torch()*self.uv_scale)
        wi_ = (flip*si.wi).torch()[...,:2]
        wo_ = (flip*wo).torch()[...,:2]
        
        f_rgb_ = self.neumip.eval_texture(uv_,wi_)
        btf_ = self.neumip.eval(f_rgb_,wi_,wo_[...,:2]).relu()
        btf_ *= (wo_[...,:2].pow(2).sum(-1,keepdim=True)<=THRESH)

        return mi.Spectrum(btf_)
    
    def pdf(self ,ctx, si, wo, active):
        flip = mi.Vector3f(1.0)
        # neumip material has flipped shading frame
        if self.flip_sh_frame:
            flip.x = dr.sign(dr.dot(si.sh_frame.s,si.dp_du))
            flip.y = dr.sign(dr.dot(si.sh_frame.t,si.dp_dv))
        
        uv_ = (si.uv.torch()*self.uv_scale)
        wi_ = (si.wi*flip).torch()[...,:2]
        wo_ = (wo*flip).torch()
        
        f_rgb_ = self.neumip.eval_texture(uv_,wi_)
        cond_ = self.neusampler.encode_cond(wi_,f_rgb_)
        
        pdf_ = self.neusampler.pdf_cond(cond_,wo_)
        
        pdf = dr.select(active,mi.Float(pdf_),0)
        return pdf
    
    def eval_pdf(self,ctx, si, wo, active=True):     
        flip = mi.Vector3f(1.0)
        # neumip material has flipped shading frame
        if self.flip_sh_frame:
            flip.x = dr.sign(dr.dot(si.sh_frame.s,si.dp_du))
            flip.y = dr.sign(dr.dot(si.sh_frame.t,si.dp_dv))
        
        uv_ = (si.uv.torch()*self.uv_scale)
        wi_ = (si.wi*flip).torch()[...,:2]
        wo_ = (wo*flip).torch()
        
        f_rgb_ = self.neumip.eval_texture(uv_,wi_)
        cond_ = self.neusampler.encode_cond(wi_,f_rgb_)
        
        pdf_ = self.neusampler.pdf_cond(cond_,wo_)
        btf_ = self.neumip.eval(f_rgb_,wi_,wo_[...,:2]).relu()
        btf_ *= (wo_[...,:2].pow(2).sum(-1,keepdim=True)<=THRESH)
        
        pdf = dr.select(active,mi.Float(pdf_),0)
        btf = mi.Spectrum(btf_)
        
        return btf,pdf
        
    def to_string(self,):
        return 'NeuBSDF'
    
mi.register_bsdf('neubsdf', lambda props: NeuBSDF(props))