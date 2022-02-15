import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y

class VIBEX(nn.Module):
    def __init__(
            self,
            seqlen,
            mean_param,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True, device='cuda',
    ):

        super(VIBEX, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = SMPLXRegressor(seqlen, mean_param)
        # regressor can predict cam, pose and shape params in an iterative way
        # self.regressor = Regressor()

    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        features = self.encoder(input)
        features = features.reshape(-1, features.size(-1))

        smplx_output = self.regressor(features) 
        # smpl_output = self.regressor(feature, J_regressor=J_regressor)
        # for s in smpl_output:
        #     s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
        #     s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
        #     s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
        #     s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
        #     s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smplx_output
class SMPLXRegressor(nn.Module):
    def __init__(self, seqlen, mean_params=None):
        super(SMPLXRegressor, self).__init__()
        self.seqlen = seqlen
        self.mean_params = mean_params
        self.fc1 = nn.Linear(512 * 4 + mean_params.shape[1], 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        #--Predict pose (318), shape (10), expression (10), camera (3)
        self.decpose = nn.Linear(1024, 318)
        self.decshape = nn.Linear(1024, 10)
        self.decexp = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decexp.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]
        _init = self.mean_params.expand(batch_size, -1)
        
        pred_pose = _init[:, :318]
        pred_shape = _init[:, 318: 328]
        pred_exp = _init[:, 328: 338]
        pred_cam = _init[:, 338: 341]
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_exp, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_exp = self.decexp(xc) + pred_exp
            pred_cam = self.deccam(xc) + pred_cam
            pred_param = torch.cat([pred_pose, pred_shape, pred_shape, pred_exp, pred_cam], 1)

        return pred_param

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=''):
        super(Regressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        # pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output
