import torch
import torch.nn as nn
from commons.keypoints import KEYPOINT_NAMES
from commons.keypoints import dset_to_body_model, get_part_idxs
class HumanLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(HumanLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none').to(device)
        self.mae = nn.L1Loss(reduction='none').to(device)
        # self.pelvis_idx = KEYPOINT_NAMES.index('pelvis')
        self.left_hip_idx = KEYPOINT_NAMES.index('left_hip')
        self.right_hip_idx = KEYPOINT_NAMES.index('right_hip')
        self.pelvis_idx = KEYPOINT_NAMES.index('pelvis')
        self.human_part = get_part_idxs()
    def denorm_kp_to_pixel(self, kp, img_size =256):
        denorm_kp = (kp[:, :] + 1.)/2  * img_size
        return denorm_kp
    def compute_kp_2d(self, x, y, visible):
        loss = visible * (self.mse(x, y))
        return loss.mean()
    def compute_2d_hands_part(self, x, y, visible):
        hand_idxs = self.human_part['hand']
        #--Remove face contour 
        #--Convert to pixel level 256 
        pred =  self.denorm_kp_to_pixel(x[:, hand_idxs])  
        gt = self.denorm_kp_to_pixel(y[:, hand_idxs])
        vis = visible[:, hand_idxs]
        loss = vis * (self.mse(pred, gt))
        return loss.mean()
    def compute_2d_head_part(self, x, y, visible):
        head_idxs = self.human_part['head']
        #--Remove face contour 
        head_idxs = head_idxs[:60]
        #--Convert to pixel level 256 
        pred =  self.denorm_kp_to_pixel(x[:, head_idxs])  
        gt = self.denorm_kp_to_pixel(y[:, head_idxs])
        vis = visible[:, head_idxs]
        loss = vis * (self.mse(pred, gt))
        return loss.mean()
    def compute_kp_3d(self, x, y, visible):
        # pred_pelvis = x[:, [self.left_hip_idx, self.right_hip_idx], :].mean(dim=1, keepdim=True)
        # gt_pelvis  = y[:, [self.left_hip_idx, self.right_hip_idx], :].mean(dim=1, keepdim=True)
        pred_pelvis = x[ :, self.pelvis_idx, :].unsqueeze(1)
        gt_pelvis = y[ :, self.pelvis_idx, :].unsqueeze(1)

        #--Remove pelvis location
        x = x - pred_pelvis
        y = y - gt_pelvis
        loss = visible * (self.mse(x, y))

        return loss.mean()
    def compute_model_smplx(self, x, y):
        loss = self.mse(x, y)
        return loss.mean()