import os
import torch
import random
from loguru import logger

import numpy as np
import os.path as osp
import joblib
from torch.utils.data import Dataset

from commons.vibe_utils import split_into_chunks, transfrom_keypoints, normalize_2d_kp
from commons.keypoints import dset_to_body_model, get_part_idxs
from commons.keyps_utils import mapping_batch_keypoints
class Dataset3D(Dataset):
    def __init__(self, dset_conf, seqlen, root_dir, overlap=0.,
                 folder=None, dataset_name=None, debug=False, phase ='train'):
        self.root_dir = root_dir
        self.phase = phase
        self.folder = folder
        self.dset_conf = dset_conf
        self.dataset_name = dset_conf['name']
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        source_idxs, target_idxs = dset_to_body_model(
            dset='vibe-spin',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=True,
            )
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        head_idxs = idxs_dict['head']
        self.indexes = {'src_idxs': self.source_idxs,
                        'tgt_idxs': self.target_idxs,
                        'body_idxs': np.asarray(body_idxs),
                        'hand_idxs': np.asarray(hand_idxs),
                        'left_hand_idxs': np.asarray(left_hand_idxs),
                        'right_hand_idxs': np.asarray(right_hand_idxs),
                        'face_idxs': np.asarray(face_idxs),
                        'head_idxs': np.asarray(head_idxs),
                        }
    def load_db(self):
        db_file = osp.join(self.root_dir, self.dataset_name, f'{self.dataset_name}_{self.phase}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        logger.success(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db
    def __len__(self):
        return len(self.vid_indices)
    def __getitem__(self, index):
        return self.get_single_item(index)
    

    def get_single_item(self, index):
        #--Load from data
        start_index, end_index = self.vid_indices[index]
        kp_2d = self.db['joints2D'][start_index:end_index + 1]
        kp_3d = self.db['joints3D'][start_index:end_index + 1]
        bbox = self.db['bbox'][start_index:end_index + 1]
        features = self.db['features'][start_index:end_index+1]
        #--Process data
        # kp_2d = mapping_batch_keypoints(kp_2d, self.indexes, self.seqlen)
        kp_2d = mapping_batch_keypoints(kp_2d, self.indexes, self.seqlen, face_contour=False)
        kp_3d = mapping_batch_keypoints(kp_2d, self.indexes, self.seqlen, face_contour=False)

        pose = np.zeros((kp_2d.shape[0], 72))
        shape = np.zeros((kp_2d.shape[0], 10))
        w_smpl = torch.zeros(self.seqlen).float()
        w_3d = torch.ones(self.seqlen).float()

        kp_2d_tensor = np.zeros((self.seqlen, 127, 3), dtype=np.float16)
        theta_tensor = np.zeros((self.seqlen, 85), dtype=np.float16)
        kp_3d_tensor = np.zeros((self.seqlen, 127, 3), dtype=np.float16)
        
        for idx in range(self.seqlen):
            kp_2d[idx,:,:2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]
        #--Change to tensor
        target = {
            'features': torch.from_numpy(features).float(),
            'theta': torch.from_numpy(theta_tensor).float(), # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(), # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(), # 3D keypoints
            'w_smpl': w_smpl,
            'w_3d': w_3d,
        }
        return target
    