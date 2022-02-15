import sys
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn

from loguru import logger
import torch.nn.functional as F

DEFAULT_FOCAL_LENGTH = 5000
def weak_persp_to_blender(
        targets,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()
    orig_bbox_size = targets['orig_bbox_size']
    bbox_center = targets['orig_center']
    z = 2 * focal_length / (camera_scale * orig_bbox_size)
    transl = [
            camera_transl[0, 0].item(), camera_transl[0, 1].item(),
            z.item()]
    shift_x = - (bbox_center[0] / W - 0.5) 
    shift_y = (bbox_center[1] - 0.5 * H) / W
    focal_length_in_mm = focal_length / W * sensor_width
    return {'focal_length_in_mm': focal_length_in_mm,
            'focal_length_in_px' : focal_length,
            'center': bbox_center,
            'transl': np.array(transl)
     }
def build_cam_proj():
    camera_scale_func = F.softplus
    mean_scale = 0.9
    mean_scale = np.log(np.exp(mean_scale) - 1)
    camera_mean = torch.tensor([mean_scale, 0.0, 0.0], dtype=torch.float32)
    camera = WeakPerspectiveCamera()
    camera_param_dim = 3
    return {
        'camera': camera,
        'mean': camera_mean,
        'scale_func': camera_scale_func,
        'dim': camera_param_dim
    }
class WeakPerspectiveCamera(nn.Module):
    ''' Scaled Orthographic / Weak-Perspective Camera
    '''

    def __init__(self):
        super(WeakPerspectiveCamera, self).__init__()

    def forward(
        self,
        points,
        scale,
        translation
    ):
        ''' Implements the forward pass for a Scaled Orthographic Camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                scale: torch.tensor, Bx1
                    The predicted scaling parameters
                translation: torch.tensor, Bx2
                    The translation applied on the image plane to the points
            Returns
            -------
                projected_points: torch.tensor, BxNx2
                    The points projected on the image plane, according to the
                    given scale and translation
        '''
        assert translation.shape[-1] == 2, 'Translation shape must be -1x2'
        assert scale.shape[-1] == 1, 'Scale shape must be -1x1'

        projected_points = scale.view(-1, 1, 1) * (
            points[:, :, :2] + translation.view(-1, 1, 2))
        return projected_points