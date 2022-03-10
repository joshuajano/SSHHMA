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
class CuratedExpose(Dataset):
    def __init__(self, cfg, human_model, phase= 'train'):
        super(CuratedExpose).__init__()
        self.human_model = human_model
        self.dset_info = cfg
        self.aug_info = cfg['augment']
        self.dset_cfg = cfg['curatedex']
        self.root_dir = self.dset_cfg['imgs_dir']
        data_path = self.dset_cfg['npz_dir']
        if phase=='train':
            data_path = os.path.join(data_path, 'train_curated_v2.npz')
        else:
            data_path = os.path.join(data_path, 'val.npz')
        data = np.load(data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        #--Extract all info inside data
        self.betas = data['betas'].astype(np.float32)
        self.expressions = data['expression'].astype(np.float32)
        self.keypoints2D = data['keypoints2D'].astype(np.float32)
        self.pose = data['pose'].astype(np.float32)
        self.cameras = data['translation'].astype(np.float32)
        self.img_fns = np.asarray(data['img_fns'], dtype=np.string_)

        #--Mapping keypoints
        source_idxs, target_idxs = dset_to_body_model(
            dset='openpose25+hands+face',
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

        self.num_items = self.pose.shape[0]
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
        self.norm = Normalize(mean=mean, std=std)
        logger.debug('Load Curated Expose dataset')
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index):
        item = {}

        pose = self.pose[index]
        shape = self.betas[index]
        exp = self.expressions[index]
        camera = self.cameras[index]
        img_fn = self.img_fns[index].decode('utf-8')
        keypoints2d = self.keypoints2D[index]
        #--Get data
        img, kp2d, pose  = self.get_data(img_fn, pose, keypoints2d)
        
        #--Make tensor
        img = torch.from_numpy(img).float()
        img = self.norm(img)
        conf = torch.from_numpy(kp2d[:, -1]).float()
        j2d = torch.from_numpy(kp2d[:, :2]).float()
        camera = torch.from_numpy(camera).float()
        shape = torch.from_numpy(shape).float()
        exp = torch.from_numpy(exp).float()
        pose = aa_SMPLX(pose)
        with torch.no_grad():
            human_mesh = self.human_model(expression = exp.unsqueeze(dim=0),
                            betas = shape.unsqueeze(dim=0),
                            global_orient = pose['global_pose'].unsqueeze(dim=0), 
                            body_pose = pose['body_pose'].unsqueeze(dim=0),
                            left_hand_pose = pose['left_hand_pose'].unsqueeze(dim=0),
                            right_hand_pose = pose['right_hand_pose'].unsqueeze(dim=0),
                            jaw_pose = pose['jaw_pose'].unsqueeze(dim=0),
                            get_skin=True, return_shaped=True, pos2rot= False)
            j3d = human_mesh.joints[0]
        #--Put in dict
        item['dset_name'] = 'curated'
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
        kp2d = mapping_keypoints(kp2d, self.indexes, 
                    self.body_idxs, self.left_hand_idxs, self.right_hand_idxs, 
                    self.face_idxs)
        
        keypoints = kp2d[:, :-1]
        conf = kp2d[:, -1]
        bbox = keyps_to_bbox(keypoints, conf, img_size=orig_shape)
        center, scale, bbox_size = bbox_to_center_scale(bbox, 
                        dset_scale_factor= self.dset_info['body_dset_factor'])
        
        img = rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        keyps = j2d_processing(kp2d, center, scale, scale, flip)
        gp = pose_processing(pose[0].copy(), rot)
        pose[0] = gp
        return img, keyps, pose 

        