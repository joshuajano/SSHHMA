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
from commons.keyps_utils import j2d_processing, j3d_processing
from commons.human_models_utils import pose_processing, convert_aa_to_rot_mat_SMPLX, aa_SMPLX
class PseudoH36M(Dataset):
    def __init__(self, cfg, phase= 'train'):
        self.dset_info = cfg
        self.aug_info = cfg['augment']
        self.dset_cfg = cfg['pseudo-h36m']
        self.root_dir = self.dset_cfg['imgs_dir']
        data_path = self.dset_cfg['npz_dir']
        #--Extract all info inside data
        self.betas, self.expressions, self.pose = [], [], []
        self.keypoints2D, self.keypoints3D, self.img_fns = [], [], []
        self.scale, self.center = [], []
        self.read_from_multiple_npz(data_path)
        self.num_items = self.center.shape[0]
        source_idxs, target_idxs = dset_to_body_model(
            dset='h36m',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour= False)
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)
        self.indexes = {
            'src_idxs': self.source_idxs, 
            'tgt_idxs': self.target_idxs
        }
        #--Get each human part index
        idxs_dict = get_part_idxs()
        self.body_idxs = np.asarray(idxs_dict['body'])
        self.hand_idxs = np.asarray(idxs_dict['hand'])
        self.left_hand_idxs = np.asarray(idxs_dict['left_hand'])
        self.right_hand_idxs = np.asarray(idxs_dict['right_hand'])
        self.face_idxs = np.asarray(idxs_dict['face'])
        self.head_idxs = np.asarray(idxs_dict['head'])
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        self.norm = Normalize(mean=mean, std=std)
        logger.debug('Load Pseudo H36M dataset')
    def __len__(self):
        return self.num_items
    def __getitem__(self, index):
        item = {}
        img_fn = self.img_fns[index].decode('utf-8')
        center = self.center[index]
        scale = self.scale[index]
        kp2d = self.keypoints2D[index]
        kp3d = self.keypoints3D[index] 
        pose = self.pose[index]
        shape = self.betas[index]
        exp = self.expressions[index]
        #--Get data
        img, keyps2d, keyps3d, pose = self.get_data(img_fn, pose, kp2d, kp3d, center, scale)

        #--Make tensor
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        conf = torch.from_numpy(keyps2d[:, -1]).float()
        j2d = torch.from_numpy(keyps2d[:, :2]).float()
        j3d = torch.from_numpy(keyps3d[:, :3]).float()
        shape = torch.from_numpy(shape).float()
        exp = torch.from_numpy(exp).float()
        pose = aa_SMPLX(pose)
        #--Put in dict
        item['dset_name'] = 'h36m'
        item['img'] = img
        item['img_fn'] = img_fn
        item['j2d'] = j2d
        item['j3d'] = j3d
        item['conf'] = conf

        #SMPL-X
        item['beta'] = shape
        item['expression'] = exp
        item['global_orient'] = pose['global_pose']
        item['body_pose'] = pose['body_pose']
        item['jaw_pose'] = pose['jaw_pose']
        item['left_hand_pose'] = pose['left_hand_pose']
        item['right_hand_pose'] = pose['right_hand_pose']
        # item['camera'] = camera
        item['gender'] = 'neutral'
        item['type'] = 0
        return item 
    def get_data (self, img_fn, pose, kp2d, kp3d, center, scale, flip=False):
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
        kp2d = mapping_keypoints(kp2d, self.indexes, 
                    self.body_idxs, self.left_hand_idxs, self.right_hand_idxs, 
                    self.face_idxs)
        kp3d = mapping_keypoints(kp3d, self.indexes, 
                    self.body_idxs, self.left_hand_idxs, self.right_hand_idxs, 
                    self.face_idxs)
        img = rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        keyps2d = j2d_processing(kp2d, center, scale, scale, flip)
        keyps3d = j3d_processing(kp3d, rot, flip)
        gp = pose_processing(pose[0].copy(), rot)
        pose[0] = gp
        return img, keyps2d, keyps3d, pose 
    def read_from_multiple_npz(self, root_dir, npz_list= ['500000', '1000000', '1500000', '1559752']):
        init = False
        for npz in npz_list:
            fn = os.path.join(root_dir, 'h36m_train_w_pseudo_SMPLX_20_'+ npz+'.npz')
            data = np.load(fn, allow_pickle=True)
            data = {key: data[key] for key in data.keys()}
            img_fns =  np.asarray(data['imgname'], dtype=np.string_) 
            scale = data['scale']
            center = data['center']
            kp2d = data['part']
            kp3d = data['S']
            betas = data['shape'].astype(np.float32)
            expressions = data['expression'].astype(np.float32)
            pose = data['pose'].astype(np.float32)
            if init :
                self.betas = np.vstack([self.betas, betas])
                self.expressions = np.vstack([self.expressions, expressions])
                self.pose = np.vstack([self.pose, pose])
                self.keypoints2D = np.vstack([self.keypoints2D, kp2d])
                self.keypoints3D = np.vstack([self.keypoints3D, kp3d])
                self.scale = np.append(self.scale, scale)
                self.center = np.vstack([self.center, center])
                self.img_fns = np.append(self.img_fns, img_fns)
            else:
                self.betas = betas
                self.expressions = expressions
                self.pose = pose
                self.keypoints2D = kp2d
                self.keypoints3D = kp3d
                self.scale = scale
                self.center = center
                self.img_fns = img_fns
                init = True
