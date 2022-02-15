import pickle
import numpy as np
import torch
from commons.priors import ContinuousRotReprDecoder
from commons.camera import build_cam_proj
def get_param_mean(pose_desc_dict):
    mean_list = []
    global_orient_mean = pose_desc_dict['global_orient']['mean']
    global_orient_mean[3] = -1
    body_pose_mean = pose_desc_dict['body_pose']['mean']
    left_hand_pose_mean = pose_desc_dict['left_hand_pose']['mean']
    right_hand_pose_mean = pose_desc_dict['right_hand_pose']['mean']
    jaw_pose_mean = pose_desc_dict['jaw_pose']['mean']
    shape_mean = pose_desc_dict['shape_mean']
    exp_mean = pose_desc_dict['exp_mean']
    camera_mean = pose_desc_dict['camera']['mean']
    return torch.cat([global_orient_mean, body_pose_mean, left_hand_pose_mean, 
                    right_hand_pose_mean, jaw_pose_mean, shape_mean, 
                    exp_mean, camera_mean]).view(1, -1)
def load_all_mean_params(mean_param_path,
        shape_mean_path, 
        num_global_orient=1,
        num_body_pose=21,
        num_rhand_pose=15,
        num_lhand_pose=15,
        num_jaw_pose=1,
        num_exp = 10, 
        num_shape = 10, dtype=torch.float32      
):
    with open(mean_param_path, 'rb') as f:
        mean_poses_dict = pickle.load(f)
    
    global_orient_desc = create_pose_param(num_global_orient)
    body_pose_desc = create_pose_param(num_body_pose, mean = mean_poses_dict['body_pose'])
    left_hand_pose_desc = create_pose_param(num_lhand_pose, mean = mean_poses_dict['left_hand_pose'])
    right_hand_pose_desc = create_pose_param(num_rhand_pose, mean = mean_poses_dict['left_hand_pose'])
    jaw_pose_desc = create_pose_param(num_jaw_pose)

    shape_mean = torch.from_numpy(np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=dtype).reshape(1, -1)[:, :num_shape].reshape(-1)
    expression_mean = torch.zeros([num_exp], dtype=dtype)
    return {
        'global_orient': global_orient_desc,
        'body_pose': body_pose_desc,
        'left_hand_pose': left_hand_pose_desc,
        'right_hand_pose': right_hand_pose_desc,
        'jaw_pose': jaw_pose_desc,
        'shape_mean' : shape_mean, 
        'exp_mean' : expression_mean,
        'camera' : build_cam_proj()
    }
def create_pose_param(num_angles, mean = None):
    decoder = ContinuousRotReprDecoder(num_angles, mean = mean)
    dim = decoder.get_dim_size()
    ind_dim = 6
    mean = decoder.get_mean()
    return {
        'decoder': decoder,
        'dim' : dim,
        'ind_dim' : ind_dim,
        'mean' : mean,
    }