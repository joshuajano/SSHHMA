import os
from site import check_enableusersite
from loguru import logger 
import torch
import pickle
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torchgeometry as tgm
from torch.optim.lr_scheduler import StepLR
# from datasets.dataset_3d import Dataset3D
from networks.holisnet import HolisNet
from datasets.curatedex import CuratedExpose
from datasets.agora import Agora
from datasets.pseudo_h36m import PseudoH36M
from datasets.mixed_dataset import MixedDataset
from datasets.threedpw_eval import ThreedpwEval

from human_models.smplxLayer import SMPLXLayer
from commons.smplx_utils import load_all_mean_params, get_param_mean
from losses.loss import HumanLoss
from commons.render import Renderer
# from commons.draw import draw_keypoints_w_black_bg
from commons.human_mesh_loader import HumanMeshLoader
from human_models.smplxLayer import SMPLXLayer
from commons.eval_utils import reconstruction_error
from commons.train_utils import check_pseudo_indexes
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
class SemiSupervisedTrainAnnotator(object):
    def __init__(self, conf, device ='cuda'):
        self.conf = conf 
        self.device = device
        #--Setup human model requirements 
        self.pose_desc_dict = load_all_mean_params(conf['all_mean_path'], conf['shape_mean_path'])
        self.mean_params = get_param_mean(self.pose_desc_dict)
        self.human_model = HumanMeshLoader(conf['smplx']['model_path'], self.pose_desc_dict)
        self.pred_human_model = SMPLXLayer(self.pose_desc_dict, conf['smplx']['model_path'])
        #--Setup network
        self.net = HolisNet(conf, self.mean_params.to(self.device))
        #-- Load pretrained network
        checkpoint = torch.load(conf['networks']['checkpoint_weight'])
        self.net.load_state_dict(checkpoint['model'])
        logger.success('success load the pretrained weights')
        self.net.to(device)
        #--Prepare dataset
        self.agora_dset = Agora(conf['datasets'])
        self.curatedExpose_dset = CuratedExpose(conf['datasets'], self.human_model.model_neutral_adult)
        self.pseudoh36m_dset  = PseudoH36M(conf['datasets'], phase='train')
        self.threedpw_valid = ThreedpwEval(conf['datasets'], phase='validation')
        #--Setup loss computation
        self.loss_w = self.conf['loss_weights']
        self.loss = HumanLoss()
        #--Setup network optimizer and scheduler
        self.optim = Adam(lr = float(conf['lr']), params=self.net.parameters(), weight_decay= 0)
        self.scheduler = StepLR(self.optim, step_size = 500, gamma = 0.9)
        #--Setup tensorboard for loss visualization
        self.writer = SummaryWriter('logs')
        self.global_step = 0
        self.max_iter = int(conf['max_inter_per_epoch'])
        #--Setup for evaluation 
        with open('data/SMPLX_to_j14.pkl', 'rb') as f:
            self.J14_regressor = pickle.load(f, encoding='latin1')
        self.SMPLX_reg_to_LSP = torch.from_numpy(self.J14_regressor).float()
        self.render = Renderer(faces = self.pred_human_model.faces)
        self.save_train_dir = 'visualize/train_with_pseudo'
        if not os.path.exists(self.save_train_dir):
            os.makedirs(self.save_train_dir)
    def __call__(self, phase ='train'):
        for epoch in range(self.conf['epochs']):
            #--Check validation with 3DPW dataset
            valid_batch_loader = DataLoader(self.threedpw_valid, batch_size=1, 
                                num_workers= 8, pin_memory= True, drop_last=True)
            valid_batch_generator = tqdm(valid_batch_loader, total = len(valid_batch_loader))
            self.validate_per_epoch(valid_batch_generator, len(valid_batch_loader))
            train_dset = {'agora': self.agora_dset, 'curatedExpose': self.curatedExpose_dset, 'h36m': self.pseudoh36m_dset}
            train_dset = MixedDataset(train_dset)
           
            train_batch_loader = DataLoader(train_dset, shuffle= True, 
                                batch_size=self.conf['batch_size'], num_workers= 8, 
                                pin_memory= True, drop_last=True)
            # train_batch_generator = tqdm(train_batch_loader, total = len(train_batch_loader))
            train_batch_generator = tqdm(train_batch_loader, total = self.max_iter)
            self.train_per_epoch(train_batch_generator)
            #--Check from 3dpw validation dataset

            self.scheduler.step()
            '''Save model'''
            torch.save({
                'model': self.net.state_dict(),
                'opt': self.optim.state_dict(),
                }, 'data/body_model_with_pseudo_h36m_20.h5')
            # torch.save({
            #     'model': self.net.state_dict(),
            #     'opt': self.optim.state_dict(),
            #     }, 'data/body_model_with_pseudo_h36m.h5')
            
    def train_per_epoch(self, batch_generator):
        self.net.train()
        num_iter = 0
        for inputs in batch_generator:
            img = inputs['img'].to(self.device)
            gt_j3d = inputs['j3d'].to(self.device)
            gt_j2d = inputs['j2d'].to(self.device)
            gt_pose_AA = torch.cat([inputs['global_orient'].unsqueeze(dim=1), inputs['body_pose'], 
                        inputs['left_hand_pose'], inputs['right_hand_pose'], 
                        inputs['jaw_pose']], dim=1).to(self.device)
            gt_shape = inputs['beta'].to(self.device)
            gt_exp = inputs['expression'].to(self.device)
            conf = inputs['conf'].unsqueeze(dim=2).to(self.device)
            pseudo_idxs, real_idxs = check_pseudo_indexes(inputs['type'].to(self.device))
            #--Feed forward
            self.optim.zero_grad()
            pred_param = self.net(img)

            #--Convert from 6D to axis-angle
            pred_dict = self.human_model.flat_body_params_to_dict(pred_param)
            pred_rot_mat = self.human_model.convert_6D_to_rot_mat(pred_dict)
            pred_mesh = self.pred_human_model.gen_smplx(pred_rot_mat)

            pred_j3d = pred_mesh.joints
            pred_j2d = self.pred_human_model.get_pred_joint2d(pred_mesh.joints, pred_rot_mat)
            pred_shape = pred_rot_mat['betas']
            pred_exp = pred_rot_mat['expression']
            pred_pose = torch.cat([pred_rot_mat['full_pose'], 
                            torch.zeros((pred_rot_mat['full_pose'].shape[0],3,1)).cuda().float()], 2)
            pred_pose_AA = tgm.rotation_matrix_to_angle_axis(pred_pose).\
                            reshape(self.conf['batch_size'], 53, 3)
            
            #--Compute loss
            loss_j2d = self.loss_w['KP_2D_W'] * self.loss_w['KP_2D_RATIO'] * self.loss.compute_kp_2d(pred_j2d, gt_j2d, conf) 
            loss_j2d_head = self.loss.compute_2d_head_part(pred_j2d, gt_j2d, conf) * 0.1
            loss_j2d_hand = self.loss.compute_2d_hands_part(pred_j2d, gt_j2d, conf) * 0.1
            loss_j3d = self.loss_w['KP_3D_W'] * self.loss_w['KP_3D_RATIO'] * self.loss.compute_kp_3d(pred_j3d, gt_j3d, conf) 

            #--Compute loss for semi-supervised approach
            if real_idxs is not None: 
                loss_real_pose_smplx = self.loss_w['POSE_W'] * \
                            self.loss.compute_model_smplx(pred_pose_AA[real_idxs], gt_pose_AA[real_idxs])
                loss_real_shape_smplx = self.loss_w['SHAPE_W'] * \
                            self.loss.compute_model_smplx(pred_shape[real_idxs], gt_shape[real_idxs])
                loss_real_exp_smplx = self.loss_w['EXP_W'] * \
                            self.loss.compute_model_smplx(pred_exp[real_idxs], gt_exp[real_idxs])
                total_loss_real_smplx = loss_real_pose_smplx + loss_real_shape_smplx + loss_real_exp_smplx
            else:
                total_loss_real_smplx = torch.FloatTensor(1).fill_(0.).to(self.device)
            if pseudo_idxs is not None:
                loss_pseudo_pose_smplx = self.loss_w['POSE_W'] * \
                            self.loss.compute_model_smplx(pred_pose_AA[pseudo_idxs], gt_pose_AA[pseudo_idxs])
                loss_pseudo_shape_smplx = self.loss_w['SHAPE_W'] * \
                            self.loss.compute_model_smplx(pred_shape[pseudo_idxs], gt_shape[pseudo_idxs])
                loss_pseudo_exp_smplx = self.loss_w['EXP_W'] * \
                            self.loss.compute_model_smplx(pred_exp[pseudo_idxs], gt_exp[pseudo_idxs])
                total_loss_pseudo_smplx = loss_pseudo_pose_smplx + loss_pseudo_shape_smplx + loss_pseudo_exp_smplx
            else:
                total_loss_pseudo_smplx = torch.FloatTensor(1).fill_(0.).to(self.device)
            # total_loss_smplx = total_loss_real_smplx
            total_loss_smplx = (self.loss_w['PSEUDO_W'] * total_loss_pseudo_smplx) + \
                                ((1 - self.loss_w['PSEUDO_W']) * total_loss_real_smplx)
            total_loss = loss_j3d + loss_j2d + loss_j2d_head + loss_j2d_hand \
                        + total_loss_smplx
            # total_loss *= 60

            total_loss.backward()
            self.optim.step()
            if self.global_step %1000 ==0:
                pred_cam = torch.stack([pred_dict['camera'][:,1],
                                    pred_dict['camera'][:,2],
                                    2* 5000./(256 * pred_dict['camera'][:,0] +1e-9)],dim=-1).detach().cpu().numpy()
                images = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
                for j in range(self.conf['batch_size']):
                    save_name = os.path.join(self.save_train_dir, f'{j}.jpg')
                    img_np = images[j].permute((1, 2, 0)).detach().cpu().numpy()
                    vert = pred_mesh.vertices[j].detach().cpu().numpy()
                    rend_img = self.render(vert, pred_cam[j], img_np)
                    rend_img *= 255.
                    cv2.imwrite(save_name, cv2.cvtColor(rend_img.astype(np.uint8), cv2.COLOR_BGR2RGB ))
            # --Write in tensorboard
            self.writer.add_scalar('Loss/3D_Keyps', loss_j3d, global_step=self.global_step)
            self.writer.add_scalar('Loss/2D_Keyps', loss_j2d, global_step=self.global_step)
            self.writer.add_scalar('Loss/2D_Head', loss_j2d_head, global_step=self.global_step)
            self.writer.add_scalar('Loss/2D_Hands', loss_j2d_head, global_step=self.global_step)
            if real_idxs is not None: 
                self.writer.add_scalar('Loss/real_pose', loss_real_pose_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/real_shape', loss_real_shape_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/real_expression', loss_real_exp_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/real_smplx', total_loss_real_smplx, global_step=self.global_step)
            if pseudo_idxs is not None:
                self.writer.add_scalar('Loss/pseudo_pose', loss_pseudo_pose_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/pseudo_shape', loss_pseudo_shape_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/pseudo_expression', loss_pseudo_exp_smplx, global_step=self.global_step)
                self.writer.add_scalar('Loss/pseudo_smplx', total_loss_pseudo_smplx, global_step=self.global_step)
            self.writer.add_scalar('Loss/total_smplx', total_loss_smplx, global_step=self.global_step)
            self.writer.add_scalar('Loss/Total', total_loss, global_step=self.global_step)
            if num_iter > 0 and num_iter % self.max_iter ==0:
                break
            self.global_step+=1
            num_iter +=1
    def validate_per_epoch(self, batch_generator, len_batch_loader):
        PA_MPJPE_scores = 0
        for inputs in batch_generator:
            self.net.eval()
            img = inputs['img'].to(self.device)
            gt_j3d = inputs['j3d'].to(self.device)
            
            gt_j3d = gt_j3d[:, H36M_TO_J14, :]
            gt_pelv = (gt_j3d[:, [2],:] + gt_j3d[:, [3],:])/2
            gt_j3d = gt_j3d - gt_pelv
            gt_j3d = gt_j3d.detach().cpu().numpy()

            with torch.no_grad():
                pred_param = self.net(img)
                #--Convert from 6D to axis-angle
                pred_dict = self.human_model.flat_body_params_to_dict(pred_param)
                pred_rot_mat = self.human_model.convert_6D_to_rot_mat(pred_dict)
                pred_mesh = self.pred_human_model.gen_smplx(pred_rot_mat)
                pred_vertices = pred_mesh.vertices
            
            reg_smplx = self.SMPLX_reg_to_LSP[None, :].expand(pred_vertices.shape[0], -1, -1).to(self.device)
            pred_j3d = torch.matmul(reg_smplx, pred_vertices)
            pred_pelv = (pred_j3d[:, [2],:] + pred_j3d[:, [3],:])/2 
            pred_j3d  = pred_j3d - pred_pelv 
            pred_j3d = pred_j3d.detach().cpu().numpy()
            PA_MPJPE_score = reconstruction_error(pred_j3d, gt_j3d, reduction=None)
            PA_MPJPE_scores +=PA_MPJPE_score
        final_scores = PA_MPJPE_scores/len_batch_loader * 1000
        logger.debug(f'PA MPJPE scores in (mm) : {final_scores}')
        return final_scores

# loss_pose_smplx = self.loss_w['POSE_W'] * self.loss.compute_model_smplx(pred_pose_AA, gt_pose_AA)
# loss_shape_smplx = self.loss_w['SHAPE_W'] * self.loss.compute_model_smplx(pred_shape, gt_shape)
# loss_exp_smplx = self.loss_w['EXP_W'] * self.loss.compute_model_smplx(pred_exp, gt_exp)
# batch_loader = DataLoader(self.agora_dset, shuffle= True, 
#                     batch_size=self.conf['batch_size'],  num_workers= 8, 
#                     pin_memory= True, drop_last=True)
# batch_loader = DataLoader(self.curatedExpose_dset, shuffle= True, 
#                     batch_size=self.conf['batch_size'],  num_workers= 8, 
#                     pin_memory= True, drop_last=True)
# train_dset = ConcatDataset([self.curatedExpose_dset, self.agora_dset, self.pseudoh36m_dset])
# train_dset = self.pseudoh36m_dset
# samples_weight = np.array([0.4, 0.2, 0.4])
# samples_weight = torch.from_numpy(samples_weight)
# sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
# train_batch_loader = DataLoader(train_dset, batch_size=self.conf['batch_size'], 
#                     num_workers= 8, sampler=sampler,
#                     pin_memory= True, drop_last=True)
# pred_cam = torch.stack([pred_dict['camera'][:,1],
#                     pred_dict['camera'][:,2],
#                     2* 5000./(256 * pred_dict['camera'][:,0] +1e-9)],dim=-1).detach().cpu().numpy()
