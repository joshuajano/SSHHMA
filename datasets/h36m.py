import cv2
import numpy as np
import os
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from commons.keypoints import dset_to_body_model, get_part_idxs
from commons.keyps_utils import mapping_keypoints
from commons.augmentation import get_aug_config
from commons.im_utils import rgb_preprocessing
from commons.bbox_utils import keyps_to_bbox, bbox_to_center_scale
from commons.keyps_utils import j2d_processing
from commons.human_models_utils import pose_processing, convert_aa_to_rot_mat_SMPLX, aa_SMPLX
class H36M(Dataset):
    def __init__(self, cfg, phase= 'test'):
        self.phase = phase
        self.dset_info = cfg
        self.aug_info = cfg['augment']
        self.dset_cfg = cfg['h36m']
        self.root_dir = self.dset_cfg['imgs_dir']
        data_path = self.dset_cfg['npz_dir']
        data_path = os.path.join(data_path, 'h36m_train.npz')
        data = np.load(data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}
        self.img_fns =  np.asarray(data['imgname'], dtype=np.string_) 
        self.scale = data['scale']
        self.center = data['center']
        self.kp2d = data['part']
        self.kp3d = data['S']
        
        self.num_items = self.center.shape[0]
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        self.norm = Normalize(mean=mean, std=std)
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index):
        item = {}
        img_fn = self.img_fns[index].decode('utf-8')
        center = self.center[index]
        scale = self.scale[index]
        kp2d = self.kp2d[index]
        kp3d = self.kp3d[index] 

        img = self.get_img_only(img_fn, center, scale)
        #--Make tensor
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        scale = torch.tensor(scale).float()
        center = torch.from_numpy(center).float()
        kp2d = torch.from_numpy(kp2d).float()
        kp3d = torch.from_numpy(kp3d).float()

        item['img'] = img
        item['img_fn'] = img_fn
        item['kp2d'] = kp2d
        item['kp3d'] = kp3d
        item['scale'] = scale
        item['center'] = center

        return item
        
    def get_img_only(self, img_fn, center, scale):
        sc, rot, flip, pn = 1, 0, 0, np.array([1, 1, 1])
        img_fn = os.path.join(self.root_dir, img_fn)
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        img = rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        return img 

