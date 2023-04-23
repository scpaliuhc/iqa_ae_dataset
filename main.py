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
MEAN=[[[0.485, 0.456, 0.406]]]
STD=[[[0.229, 0.224, 0.225]]]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # device
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # dataset
    DL=load_dataset(cfg)
    
    # model
    TM=load_model(cfg)
    class Normalize(nn.Module) :
        def __init__(self, mean, std) :
            super(Normalize, self).__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))
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

def load_model(cfg):
    return getattr(models,cfg.Target_Models[cfg.TM_ID])(pretrained=True)

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
                   device=device)
    elif cfg.Methods[cfg.ME_ID] == "SparseFool":
        atk_=atk(model,
                 c=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                 steps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "CDP":
        from src.CDP import CDP
        atk_=CDP(device=device,
                    eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                    checkpoint=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "AdvGAN":
        from src.advGAN import advGAN
        atk_=advGAN(device,
                    eps=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][0],
                    checkpoint=cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID][1])
    elif cfg.Methods[cfg.ME_ID] == "AdvPatch":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "Square":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "UAP":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "GAP":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "FFF":
        raise NotImplementedError
    elif cfg.Methods[cfg.ME_ID] == "GUAP":
        raise NotImplementedError
    return atk_

def carry_attack(atk_,dataset,device,save_dir,cfg,model):
    desc=f"{cfg.Methods[cfg.ME_ID]} {cfg.Parameters[cfg.Methods[cfg.ME_ID]][cfg.PA_ID]}"
    dataset=tqdm(dataset,ncols=100,desc=desc)
    id=0
    for img,file in dataset:
        if id>cfg.Test_Amount:
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
        else:
            success='F'
       
        full_name = f"{pic_name}_{str(cfg.TM_ID).zfill(2)}_{str(cfg.ME_ID).zfill(2)}_{str(cfg.PA_ID).zfill(2)}_{success}_{fake_label[0]}_{pred_label[0]}.png"
        
        img_adv=img_adv[0].cpu().numpy().transpose(1,2,0)
        imageio.imsave(os.path.join(save_dir,full_name),img_as_ubyte(img_adv))
        
if __name__=='__main__':
    main()