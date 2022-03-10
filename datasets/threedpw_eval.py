import cv2
import numpy as np
import os
import torch.utils.data as dutils
from torchvision.transforms import Normalize
import torch
from loguru import logger

from human_models.smplLayer import SMPL
from commons.keypoints import dset_to_body_model, get_part_idxs
from commons.keyps_utils import mapping_keypoints
from commons.im_utils import rgb_preprocessing
from commons.bbox_utils import keyps_to_bbox, bbox_to_center_scale
from commons.keyps_utils import j2d_processing
from commons.human_models_utils import pose_processing, convert_aa_to_rot_mat_SMPL, aa_SMPLX

class ThreedpwEval(dutils.Dataset):
    def __init__(self, cfg, phase='test'):
        super(ThreedpwEval, self).__init__()
        dset_conf = cfg['3dpw-eval']
        if phase=='test':
            data_path = os.path.join(dset_conf['npz_dir'], '3dpw_test.npz')
        elif phase=='validation':
            data_path = os.path.join(dset_conf['npz_dir'], '3dpw_valid.npz')
        data = np.load(data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()} 
        self.root_dir = dset_conf['imgs_dir']
        self.imgs_name = np.asarray(data['imgname'], dtype=np.string_)
        # self.smplx = self.data['smplx']
        self.poses = data['pose']
        self.shapes = data['shape']
        self.scales = data['scale']
        self.centers = data['center']
        self.genders = data['gender']
        
        self.len_data = len(self.imgs_name)
        mean = [0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        
        self.norm = Normalize(mean=mean, std=std)
        self.smpl_neutral = SMPL('data/human_models/smpl/', gender='neutral', create_transl=False)
        self.smpl_male = SMPL('data/human_models/smpl/', gender='male', create_transl=False)
        self.smpl_female = SMPL('data/human_models/smpl/', gender='female', create_transl=False)
        self.SMPL_J_regressor = torch.from_numpy(np.load('data/J_regressor_h36m.npy')).float()
        if phase=='test':
            logger.debug('Load 3DPW test dataset ')
        elif phase=='validation':
            logger.debug('Load 3DPW validation dataset ')
    def __len__(self):
        return self.len_data
    def get_data(self, img_fn, center, scale):
        flip = 0            
        pn = np.ones(3) 
        rot = 0
        sc = 1 
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy()
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]

        img = rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        return img
    def __getitem__(self, index):
        item = {}

        '''------------Load from list-----------------'''
        img_fn = self.imgs_name[index].decode('utf-8')
        pose = self.poses[index]
        shape = self.shapes[index]
        center = self.centers[index]
        scale = self.scales[index]

        '''--------------Process data------------------'''
        img_fn = os.path.join(self.root_dir, img_fn)
        shape = np.array(shape).astype(np.float32)
        pose = np.array(pose).astype(np.float32)
        raw_pose = pose
        pose = pose.reshape(-1, 3)
        gp = pose[0]
        
        img = self.get_data(img_fn, center, scale)

        '''------------Make it tensor--------------'''
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        pose_param = convert_aa_to_rot_mat_SMPL(gp, pose)
        shape = torch.from_numpy(shape).float()
        pose = torch.from_numpy(pose).float()
        raw_pose = torch.from_numpy(raw_pose).float()
        shape = shape.unsqueeze(dim=0)
        body_pose = pose_param['body_pose'].unsqueeze(dim=0)
        global_orient = pose_param['global_pose'].unsqueeze(dim=0)
        #--Get SMPL from GT
        if self.genders[index][0] =='m':
            human_mesh = self.smpl_male(betas=shape, body_pose=body_pose, 
                                    global_orient=global_orient, pose2rot=False)
        elif self.genders[index][0] =='f':
            human_mesh = self.smpl_female(betas=shape, body_pose=body_pose, 
                                    global_orient=global_orient, pose2rot=False)
        else:
            human_mesh = self.smpl_neutral(betas=shape, body_pose=body_pose, 
                                    global_orient=global_orient, pose2rot=False)
        vertices = human_mesh.vertices
        reg_smpl = self.SMPL_J_regressor[None, :].expand(vertices.shape[0], -1, -1)
        lsp_14_joints = torch.matmul(reg_smpl, vertices)
        item['img'] = img   
        item['j3d'] = lsp_14_joints[0]
        # item['shape'] = shape
        # item['global_orient'] = pose_param['global_pose']
        # item['body_pose'] = pose_param['body_pose']
        # item['raw_pose'] = raw_pose
        item['gender'] = self.genders[index][0]
        return item 