from src.tools import Normalize
from art.attacks.evasion import *
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

class AdvPatch_v1(object): #效果不好
    def __init__(self,model,device) -> None:
        self.model=nn.Sequential(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],device=device),model)
        self.model_art=PyTorchClassifier(model=self.model,input_shape=[3,299,299],loss=nn.CrossEntropyLoss(),optimizer=optim.Adam(self.model.parameters(),lr=0.01),nb_classes=1000)
        self.attack=AdversarialPatchPyTorch(estimator=self.model_art,patch_shape=(3,299,299),batch_size=1,patch_type='square',optimizer='Adam',learning_rate=5,max_iter=1000,targeted=False,distortion_scale_max=0,rotation_max=40)
    def __call__(self, img,label):
        (patch,mask)=self.attack.generate(x=img,y=label)
        # mask=mask.transpose((1,2,0))
        # mask=cv2.resize(mask,(299,299))
        # mask=mask.transpose((2,0,1)).astype(bool)
        mask=np.reshape(mask[0,:,:],(1,299,299)).astype(bool)
        # print(f'!!!!{patch.shape}{mask.shape}')
        adv_img=self.attack.apply_patch(img,0.4,patch,mask)
        adv_img=torch.clamp(adv_img,0.,1.)
        return adv_img.detach()
    
class AdvPatch_v2(object):
    def __init__(self,model,patch_type,target,patch,device,mean,std) -> None:
        self.model=model
        self.patch_type=patch_type
        self.target=target
        self.patch=patch
        self.device=device
        self.mean=mean
        self.std=std
    def __call__(self, x, y):
        x=(x-self.mean)/self.std
        applied_patch, mask, x_location, y_location = mask_generation(self.patch_type, self.patch, image_size=(3, 299, 299))
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), x.type(torch.FloatTensor))
        perturbated_image = perturbated_image.to(self.device)
        perturbated_image = torch.clamp(perturbated_image*self.std+self.mean,0,1)
        return perturbated_image
        

def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 299, 299)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location