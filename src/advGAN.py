import torch.nn as nn
import torch
import torchvision.transforms as transforms
class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc, target='Auto'):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        if target == 'HighResolution':
            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, image_nc, kernel_size=5, stride=1, padding=0, bias=False),
                # nn.Tanh()
                # state size. image_nc x 28 x 28
            ]
        else:
            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
                nn.Tanh()
                # state size. image_nc x 28 x 28
            ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        # self.norm_layer=Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, x):
        # x = self.norm_layer(x)
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to('cuda'))
        self.register_buffer('std', torch.Tensor(std).to('cuda'))
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class NormalizeInverse(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        invert_norm = transforms.Normalize(
            mean=[-m_elem/s_elem for m_elem, s_elem in zip(self.mean, self.std)], 
            std=[1/elem for elem in self.std]
        )
        return invert_norm(tensor)
    
class advGAN(object):
    def __init__(self,device,eps,checkpoint) -> None:
        self.generator=Generator(3,3,'HighResolution').to(device)
        self.generator.load_state_dict(torch.load(checkpoint))
        self.generator.eval()
        self.norm_layer=Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.eps=eps
        self.inv_norm=NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __call__(self, img, label):
        img=self.norm_layer(img)
        perturbation = self.generator(img)
        adv_img = torch.clamp(perturbation, -self.eps, self.eps) + img
        adv_img = torch.clamp(adv_img, 0, 1)

        # adv_img = torch.clamp(perturbation, -self.eps, self.eps) + img
        # adv_img = torch.clamp(adv_img, 0, 1)
        return adv_img.detach()
