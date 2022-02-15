import numpy as np
import cv2
import torch

def convert_6D_to_axis_angle(param):
    
    pass
def aa_SMPLX(pose):
    eye_offset = 0 if pose.shape[0] == 53 else 2
    global_pose = torch.from_numpy(pose[0])
    body_pose = torch.from_numpy(pose[1:22, :])
    jaw_pose = torch.from_numpy(pose[22])
    jaw_pose = jaw_pose.unsqueeze(dim=0)
    left_hand_pose = torch.from_numpy(pose[23 + eye_offset:23 + eye_offset + 15])
    right_hand_pose = torch.from_numpy(pose[23 + 15 + eye_offset:])
    return { 'global_pose': global_pose,
             'body_pose': body_pose,
             'jaw_pose': jaw_pose,
             'left_hand_pose': left_hand_pose,
             'right_hand_pose': right_hand_pose}
def convert_aa_to_rot_mat_SMPL(global_orient, pose):
    global_pose =  torch.from_numpy(global_orient)
    global_pose = global_pose.unsqueeze(dim=0)
    body_pose = torch.from_numpy(pose[1:24, :])

    # use batch rodrigues
    global_pose = batch_rodrigues(global_pose.view(-1, 3)).view(1, 3, 3)
    body_pose = batch_rodrigues(body_pose.view(-1, 3)).view(23, 3, 3)
    
    return { 'global_pose': global_pose,
             'body_pose': body_pose,
    }
def convert_aa_to_rot_mat_SMPLX(pose):
    eye_offset = 0 if pose.shape[0] == 53 else 2
    global_pose = torch.from_numpy(pose[0])
    body_pose = torch.from_numpy(pose[1:22, :])
    jaw_pose = torch.from_numpy(pose[22])
    jaw_pose = jaw_pose.unsqueeze(dim=0)
    left_hand_pose = torch.from_numpy(pose[23 + eye_offset:23 + eye_offset + 15])
    right_hand_pose = torch.from_numpy(pose[23 + 15 + eye_offset:])

    # use batch rodrigues
    global_pose = batch_rodrigues(global_pose.view(-1, 3)).view(1, 3, 3)
    body_pose = batch_rodrigues(body_pose.view(-1, 3)).view(21, 3, 3)
    jaw_pose = batch_rodrigues(jaw_pose.view(-1, 3)).view(1, 3, 3)
    left_hand_pose = batch_rodrigues(left_hand_pose.view(-1, 3)).view(15, 3, 3)
    right_hand_pose = batch_rodrigues(right_hand_pose.view(-1, 3)).view(15, 3, 3)
    return { 'global_pose': global_pose,
             'body_pose': body_pose,
             'jaw_pose': jaw_pose,
             'left_hand_pose': left_hand_pose,
             'right_hand_pose': right_hand_pose}

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)
def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat
def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa
def pose_processing(pose, r, f = 0):
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], r)
    # flip the pose parameters
    if f:
        pose = flip_pose(pose)
    # (72),float
    pose = pose.astype('float32')
    return pose