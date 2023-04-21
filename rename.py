import os
import pickle
import imageio
from tqdm import tqdm
import json
dic={} #"new_name":"old_name"
root='./REF-V1'
id=0
for pic in tqdm(os.listdir(root)):
    new_name=str(id).zfill(4)+'.png'
    try:
        img=imageio.imread(os.path.join(root,pic))
        id+=1
    except:
         os.remove(os.path.join(root,pic))
         continue
    imageio.imsave(os.path.join(root,new_name),img)
    os.remove(os.path.join(root,pic))
    dic[new_name]=pic
    info=json.dumps(dic,sort_keys=False,indent=4, separators=(',', ': '))
    f = open('rename.json', 'w')
    f.write(info)
    
