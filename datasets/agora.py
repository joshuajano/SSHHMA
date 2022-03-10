import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from loguru import logger
from commons.keypoints import dset_to_body_model, get_part_idxs
from commons.keyps_utils import mapping_keypoints
from commons.augmentation import get_aug_config
from commons.im_utils import rgb_preprocessing
from commons.bbox_utils import keyps_to_bbox, bbox_to_center_scale
from commons.keyps_utils import j2d_processing
from commons.human_models_utils import pose_processing, convert_aa_to_rot_mat_SMPLX, aa_SMPLX

class Agora(Dataset):
    def __init__(self, cfg, dset_name='agora', phase ='train'):
        super().__init__()
        self.dset_info = cfg
        self.dset_cfg = cfg[dset_name]
        self.aug_info = cfg['augment']

        data_path = self.dset_cfg['npz_dir']
        data_path = os.path.join(data_path, 'agora_wo_kids.npz')
        data = np.load(data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}
        self.root_dir = os.path.join(self.dset_cfg['imgs_dir'], phase)  
        self.keypoints2D = data['joints2d'].astype(np.float32)
        self.keypoints3D = data['joints3d'].astype(np.float32)
        self.vertices = data['vertices'].astype(np.float32)
        self.genders = np.asarray(data['gender'], dtype=np.string_)
        self.img_fns = np.asarray(data['imgname'], dtype=np.string_)
        #--Extract all info inside data
        self.transl = data['transl'].astype(np.float32)
        self.shapes = data['shapes'].astype(np.float32)
        self.expressions = data['expression'].astype(np.float32)

        self.global_orients = data['global_orients'].astype(np.float32)
        self.body_poses = data['body_poses'].astype(np.float32)
        self.lhand_poses = data['left_hand_poses'].astype(np.float32)
        self.rhand_poses = data['right_hand_poses'].astype(np.float32)
        self.jaw_poses = data['jaw_poses'].astype(np.float32)
        self.leye_poses = data['leye_poses'].astype(np.float32)
        self.reye_poses = data['reye_poses'].astype(np.float32)
        
        self.num_items = self.body_poses.shape[0]
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        self.norm = Normalize(mean=mean, std=std)
        logger.debug('Load AGORA dataset ')
    def __len__(self):
        return self.num_items
    def __getitem__(self, index):
        item = {}
        img_fn = self.img_fns[index].decode('utf-8')
        # gender = self.genders[index].decode('utf-8')
        shape = self.shapes[index]
        exp = self.expressions[index]
        global_orient = self.global_orients[index]
        body_pose = self.body_poses[index]
        lhand_pose = self.lhand_poses[index]
        rhand_pose = self.rhand_poses[index]
        jaw_pose = self.jaw_poses[index]

        keypoints2d = self.keypoints2D[index]
        keypoints3d = self.keypoints3D[index]
        vertices = self.vertices[index]

        pose = np.vstack(
            [global_orient, body_pose.reshape(21, 3), 
            lhand_pose.reshape(15, 3), rhand_pose.reshape(15, 3), jaw_pose
        ])
        img, kp2d, pose =self.get_data(img_fn, pose, keypoints2d)
        
        #--Make tensor
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        conf = torch.from_numpy(kp2d[:, -1]).float()
        j2d = torch.from_numpy(kp2d[:, :2]).float()
        j3d = torch.from_numpy(keypoints3d).float()
        shape = torch.from_numpy(shape).float()
        exp = torch.from_numpy(exp).float()
        pose = aa_SMPLX(pose)
        #--Put in dict
        item['dset_name'] = 'agora'
        item['img'] = img
        item['img_fn'] = img_fn
        item['j2d'] = j2d
        item['j3d'] = j3d
        item['conf'] = conf

        #SMPL-X
        item['beta'] = shape[0]
        item['expression'] = exp[0]
        item['global_orient'] = pose['global_pose']
        item['body_pose'] = pose['body_pose']
        item['jaw_pose'] = pose['jaw_pose']
        item['left_hand_pose'] = pose['left_hand_pose']
        item['right_hand_pose'] = pose['right_hand_pose']
        item['gender'] = 'neutral'
        item['type'] = 1
        return item
    def get_data (self, img_fn, pose, kp2d, flip=False):
        sc, rot, pn = get_aug_config(
                        self.aug_info['scale_factor'],
                        self.aug_info['rot_factor'],
                        self.aug_info['color_factor'])
        img_fn = os.path.join(self.root_dir, img_fn)
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        conf = np.ones((len(kp2d), 1))
        for kp in range(len(kp2d)):
            if kp2d[kp][0] >= 1280 or kp2d[kp][1] >= 720:
                conf[kp] =0
        kp2d_conf = np.hstack([kp2d, conf])
        keypoints = kp2d
        bbox = keyps_to_bbox(keypoints, conf[:, 0], img_size=orig_shape)
        center, scale, bbox_size = bbox_to_center_scale(bbox, 
                        dset_scale_factor= self.dset_info['body_dset_factor'])
        img = rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        keyps = j2d_processing(kp2d_conf, center, scale, scale, flip)
        gp = pose_processing(pose[0].copy(), rot)
        pose[0] = gp
        return img, keyps, pose