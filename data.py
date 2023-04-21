from torch.utils.data import dataset
from torchvision import transforms
import os
from PIL import Image
from logzero import logger as log
import imageio
from skimage import img_as_ubyte

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class IMGData(dataset.Dataset):
    def __init__(self,size,path) -> None:
        super(IMGData,self).__init__()
        self.tran=transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                ])
        self.files=[os.path.join(path,x) for x in os.listdir(path) if self.is_image_file(x)]

    def __getitem__(self, index):
        img=self.load_img(self.files[index])
        img=self.tran(img)
        try:
            assert list(img.shape)[0] == 3
        except:
            # log.debug(f'{self.files[index]},{img.shape}')
            img=img.repeat(3,1,1)
            over_img=img.cpu().numpy().transpose((1,2,0))
            log.debug(f'{self.files[index]},{over_img.shape}')
            imageio.imsave(os.path.join(self.files[index]), img_as_ubyte(over_img))
        return img,self.files[index]

    def __len__(self):
        return len(self.files)

    def load_img(self,filepath):
        img = Image.open(filepath)
        return img

    def is_image_file(self,filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)