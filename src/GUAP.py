import numpy as np
import torch

class GUAP(object):
    def __init__(self,flow_field,noise,mean,std,device,mode) -> None:
        self.flow_field=flow_field
        self.noise=noise
        self.device=device
        self.mean=mean
        self.std=std
        self.mode=mode

    def __call__(self, x, y):

        X_st = flow_st(x,self.flow_field)
        
        X_noise = torch.clamp(x+self.noise,0,1)
        
        X_perb = X_st+ self.noise
        X_perb = torch.clamp(X_perb, 0, 1)

        if self.mode=='st':
            return torch.clamp(X_st,0,1).detach()
        elif self.mode=='noise':
            return X_noise.detach()
        elif self.mode=='perb':
            return X_perb.detach()


def flow_st(images, flows):

    batch_size,_,H,W = images.size()

    device = images.device


    # basic grid: tensor with shape (2, H, W) with value indicating the
    # pixel shift in the x-axis or y-axis dimension with respect to the
    # original images for the pixel (2, H, W) in the output images,
    # before applying the flow transforms
    grid_single = torch.stack(
                torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
            ).float()


    grid = grid_single.repeat(batch_size, 1, 1, 1)#100,2,28,28

    images = images.permute(0,2,3,1) #100, 28,28,1

    grid = grid.to(device)

    grid_new = grid + flows
    # assert 0

    sampling_grid_x = torch.clamp(
        grid_new[:, 1], 0., (W - 1.)
            )
    sampling_grid_y = torch.clamp(
        grid_new[:, 0], 0., (H - 1.)
    )
    
    # now we need to interpolate

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a square around the point of interest
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate image boundaries
    # - 2 for x0 and y0 helps avoiding black borders
    # (forces to interpolate between different points)
    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)


    b =torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, H, W).to(device)
    # assert 0 
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()


    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    perturbed_image = wa * Ia+ wb * Ib+ wc * Ic+wd * Id
 

    perturbed_image = perturbed_image.permute(0,3,1,2)

    return perturbed_image

