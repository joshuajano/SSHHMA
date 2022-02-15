import torch
from torchvision.transforms import Normalize
import cv2
import numpy as np
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]
norm = Normalize(mean=MEAN, std=STD)
def get_demo_input(img_fn, keypoints=None, IMG_RES = 256):
    img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
    if keypoints is None:
        img = cv2.resize(img, tuple([IMG_RES, IMG_RES]), interpolation=cv2.INTER_LINEAR)
    else:
        pass
    img = np.transpose(img.astype('float32'),(2,0,1))/255.0
    img = torch.from_numpy(img).float()
    img_tensor = norm(img)
    return img_tensor.unsqueeze(dim=0)