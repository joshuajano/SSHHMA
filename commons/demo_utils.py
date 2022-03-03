import torch
from torchvision.transforms import Normalize
import cv2
import numpy as np

from commons.bbox_utils import keyps_to_bbox, bbox_to_center_scale
from commons.im_utils import crop
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]
norm = Normalize(mean=MEAN, std=STD)
def get_demo_input_using_keyp_RCNN(img_fn, rcnn_model, device='cuda'):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    img = np.transpose(img.astype('float32'),(2,0,1))/255.0
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(dim=0)
    img = img.to(device)
    output = rcnn_model(img)
    #--Please note that we assume the image only has a single person
    bbox = output[0]['boxes'].detach().cpu().numpy()
    center, scale, bbox_size = bbox_to_center_scale(bbox, 
                        dset_scale_factor= 1.2)
    img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
    crop_img = crop(img, center, scale, [256, 256], rot=0)
    crop_img = np.transpose(crop_img.astype('float32'),(2,0,1))/255.0
    crop_img = torch.from_numpy(crop_img).float()
    crop_img_tensor = norm(crop_img)
    # cv2.imwrite('test.jpg', crop_img)
    return crop_img_tensor.unsqueeze(dim=0)

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