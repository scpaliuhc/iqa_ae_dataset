import hydra
import torch
from omegaconf import DictConfig
from data import IMGData
from torchvision import models
from torch.utils.data import DataLoader
import torchattacks
import torch.nn as nn
# from torchattacks import *
import os
from tqdm import tqdm
from logzero import logger as log
import imageio
# import cv2
from skimage import img_as_ubyte
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
global device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
MEAN=torch.tensor([0.485, 0.456, 0.406]).view((1,3,1,1)).to(device)
STD=torch.tensor([0.229, 0.224, 0.225]).view((1,3,1,1)).to(device)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # dataset
    DL=load_dataset(cfg)
    
    # model
    TM=load_model(cfg,device)
    
    # make dir ae/method/parameter
    save_dir=os.path.join('./AE',str(cfg.ME_ID),str(cfg.PA_ID))

    try:
        os.makedirs(save_dir)
    except:
        None

    # attack
    log.debug(f'\ndevice: {device}\nsave_dir: {save_dir}\nTM: {cfg.Target_Models[cfg.TM_ID]}')
    atk_=load_attack(cfg,TM)
    log.debug(f'use {atk_} to attack')
    carry_attack(atk_,DL,device,save_dir,cfg,TM)


def load_dataset(cfg):
    dataset=IMGData(cfg.Image_Size,cfg.Image_Dir)
    dataloader=DataLoader(dataset,1,shuffle=False,drop_last=False,num_workers=2)
    return dataloader

def load_model(cfg,device):
    TM = getattr(models,cfg.Target_Models[cfg.TM_ID])(pretrained=True)
    class Normalize(nn.Module) :
        def __init__(self, mean, std) :
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean).to(device))
            self.register_buffer('std', torch.Tensor(std).to(device))
        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    TM = nn.Sequential(
        norm_layer,
        TM
    )
    TM=TM.to(device).eval()
    return TM

def load_attack(cfg,model):
    if cfg.Methods[cfg.ME_ID] in torchattacks.__all__:
        atk=getattr(torchattacks,cfg.Methods[cfg.ME_ID])
        
    if cfg.Methods[cfg.ME_ID] == 'FGSM':
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0])
    elif cfg.Methods[cfg.ME_ID] == 'BIM':
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 alpha=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2])
    elif cfg.Methods[cfg.ME_ID] == 'PGD':
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 alpha=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 random_start=True)
    elif cfg.Methods[cfg.ME_ID] == 'MIFGSM':
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 alpha=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 decay=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][3])
    elif cfg.Methods[cfg.ME_ID] == 'NES':
        from src.NES import NES
        atk_=NES(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 alpha=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 device=device)
    elif cfg.Methods[cfg.ME_ID] == 'CW':
        atk_=atk(model,
                 c=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == 'SimBA':
        from src.SimBA import SimBA
        atk_=SimBA(model,
                   eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                   steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                   early_stop=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                   device=device)
    elif cfg.Methods[cfg.ME_ID] == "SparseFool":
        atk_=atk(model,
                 lam=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "CDP":
        from src.CDP import CDP
        atk_=CDP(device=device,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 checkpoint=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "AdvGAN":
        from src.advGAN import AdvGAN
        atk_=AdvGAN(device=device,
                    eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                    checkpoint=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "AdvPatch":
        from src.AdvPatch import AdvPatch_v2
        with open(cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],'rb') as f:
            import pickle
            patch=pickle.load(f)
        atk_=AdvPatch_v2(model,
                         device=device,
                         patch_type='rectangle',
                         target=int(cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0].split('_')[-1]),
                         patch=patch,
                         mean=MEAN,
                         std=STD,
                         )
    elif cfg.Methods[cfg.ME_ID] == "Square":
        atk_=atk(model, 
                 norm=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 p_init=0.8,#cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 n_queries=1000)
    elif cfg.Methods[cfg.ME_ID] == "UAP":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "GAP":
        from src.GAP import GAP
        atk_=GAP(model,
                 device=device,
                 mean=MEAN,
                 std=STD,
                 mode=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 checkpoint=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 )
    elif cfg.Methods[cfg.ME_ID] == "FFF":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "GUAP":
        from src.GUAP import GUAP
        import numpy as np
        flow=np.load(cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0])
        flow=torch.tensor(flow).to(device)
        noise=np.load(cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
        noise=torch.tensor(noise).to(device)
        atk_=GUAP(flow,noise,MEAN,STD,device,cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2])
    elif cfg.Methods[cfg.ME_ID] == "DeepFool":
        atk_=atk(model,steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],overshoot=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "DIFGSM":
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 alpha=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 decay=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][3],
                 )
    elif cfg.Methods[cfg.ME_ID] == "SPSA":
        atk_=atk(model,
                 eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 delta=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 nb_iter=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2],
                 nb_sample=100,
                 max_batch_size=5,
                 )
    elif cfg.Methods[cfg.ME_ID] == "Pixle":
        atk_=atk(model,
                 x_dimensions=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 restarts=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1],
                 max_iterations=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][2]
                 )
    else:
        raise ValueError
    return atk_

def carry_attack(atk_,dataset,device,save_dir,cfg,model):
    desc=f"{cfg.Methods[cfg.ME_ID]} {cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID]}"
    dataset=tqdm(dataset,ncols=80,desc=desc)
    id=0
    count=0
    for img,file in dataset:
        if id>=cfg.Test_Amount:
            break
        id+=1
        img = img.to(device)
        _, pic_name = file,os.path.split(file[0])
        pic_name = pic_name[1].split('.')[0]

        pred_before = model(img)
        fake_label = torch.argmax(pred_before,keepdim=True)
        fake_label = fake_label.view([-1])

        img_adv=atk_(img,fake_label)
        pred_after = model(img_adv)
        pred_label = torch.argmax(pred_after,keepdim=True)
        pred_label = pred_label.view([-1])

        if fake_label[0] != pred_label[0]:
            success='T'
            count+=1
        else:
            success='F'
       
        full_name = f"{pic_name}_{str(cfg.TM_ID).zfill(2)}_{str(cfg.ME_ID).zfill(2)}_{str(cfg.PA_ID).zfill(2)}_{success}.png"
        # _{fake_label[0]}_{pred_label[0]}.png"
        
        img_adv=img_adv[0].cpu().numpy().transpose(1,2,0)
        imageio.imsave(os.path.join(save_dir,full_name),img_as_ubyte(img_adv))
    log.debug(f'attack success rate {count/id:.4f}')

if __name__=='__main__':
    main()