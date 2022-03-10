import os
from loguru import logger 
import torch
import pickle
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torchgeometry as tgm
from torch.optim.lr_scheduler import StepLR
# from datasets.dataset_3d import Dataset3D
from networks.holisnet import HolisNet
from datasets.curatedex import CuratedExpose
from datasets.agora import Agora
from datasets.threedpw_eval import ThreedpwEval

from human_models.smplxLayer import SMPLXLayer
from commons.smplx_utils import load_all_mean_params, get_param_mean
from losses.loss import HumanLoss
from commons.render import Renderer
# from commons.draw import draw_keypoints_w_black_bg
from commons.human_mesh_loader import HumanMeshLoader
from human_models.smplxLayer import SMPLXLayer
from commons.eval_utils import reconstruction_error
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
class TrainAnnotator(object):
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
        self.net.to(device)
        
        #--Prepare dataset
        self.threedpw_eval_dset = ThreedpwEval(conf['datasets'])
        self.agora_dset = Agora(conf['datasets'])
        self.curatedExpose_dset = CuratedExpose(conf['datasets'], self.human_model.model_neutral_adult)
        #--Setup loss computation
        self.loss_w = self.conf['loss_weights']
        self.loss = HumanLoss()
        #--Setup network optimizer and scheduler
        self.optim = Adam(lr = float(conf['lr']), params=self.net.parameters(), weight_decay= float(conf['lr_decay']))
        self.scheduler = StepLR(self.optim, step_size = 500, gamma = 0.9)
        #--Setup tensorboard for loss visualization
        self.writer = SummaryWriter('logs')
        self.global_step = 0
        
        #--Setup for evaluation 
        with open('data/SMPLX_to_j14.pkl', 'rb') as f:
            self.J14_regressor = pickle.load(f, encoding='latin1')
        self.SMPLX_reg_to_LSP = torch.from_numpy(self.J14_regressor).float()
        self.render = Renderer(faces = self.pred_human_model.faces)
        self.save_train_dir = 'visualize/train'
        if not os.path.exists(self.save_train_dir):
            os.makedirs(self.save_train_dir)
    def __call__(self, phase ='train'):
        if phase =='eval':
            logger.debug('Evaluation mode on 3DPW')
            eval_batch_loader = DataLoader(self.threedpw_eval_dset, shuffle= False, 
                                    batch_size=1,  num_workers= 2, 
                                    pien_memory= True, drop_last=True)
            eval_batch_generator = tqdm(eval_batch_loader, total = len(eval_batch_loader))
            self.eval_per_epoch(eval_batch_generator, len(eval_batch_loader))
        else:
            for epoch in range(self.conf['epochs']):
                train_dset = ConcatDataset([self.curatedExpose_dset, self.agora_dset])
                # if epoch%5 ==0:
                #     eval_batch_loader = DataLoader(self.threedpw_eval_dset, shuffle= False, 
                #                         batch_size=1,  num_workers= 2, 
                #                         pien_memory= True, drop_last=True)
                #     eval_batch_generator = tqdm(eval_batch_loader, total = len(eval_batch_loader))
                #     self.eval_per_epoch(eval_batch_generator, len(eval_batch_loader))
                train_batch_loader = DataLoader(train_dset, shuffle= True, 
                                    batch_size=self.conf['batch_size'], num_workers= 8, 
                                    pin_memory= True, drop_last=True)
                train_batch_generator = tqdm(train_batch_loader, total = len(train_batch_loader))
                self.train_per_epoch(train_batch_generator, len(train_batch_loader))
                self.scheduler.step()
                '''Save model'''
                torch.save({
                    'model': self.net.state_dict(),
                    'opt': self.optim.state_dict(),
                    }, 'data/body_model.h5')
                # batch_loader = DataLoader(self.agora_dset, shuffle= True, 
                #                     batch_size=self.conf['batch_size'],  num_workers= 8, 
                #                     pin_memory= True, drop_last=True)
                # batch_loader = DataLoader(self.curatedExpose_dset, shuffle= True, 
                #                     batch_size=self.conf['batch_size'],  num_workers= 8, 
                #                     pin_memory= True, drop_last=True)
            
            
    def train_per_epoch(self, batch_generator, len_batch_loader):
        self.net.train()
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
            loss_j2d = self.loss_w['KP_2D_W'] * self.loss.compute_kp_2d(pred_j2d, gt_j2d, conf)
            loss_j2d_head = self.loss.compute_2d_head_part(pred_j2d, gt_j2d, conf) * 0.1
            loss_j2d_hand = self.loss.compute_2d_hands_part(pred_j2d, gt_j2d, conf) * 0.1
            loss_j3d = self.loss_w['KP_3D_W'] * self.loss_w['KP_3D_RATIO'] * self.loss.compute_kp_3d(pred_j3d, gt_j3d, conf) 
            loss_pose_smplx = self.loss_w['POSE_W'] * self.loss.compute_model_smplx(pred_pose_AA, gt_pose_AA)
            loss_shape_smplx = self.loss_w['SHAPE_W'] * self.loss.compute_model_smplx(pred_shape, gt_shape)
            loss_exp_smplx = self.loss_w['EXP_W'] * self.loss.compute_model_smplx(pred_exp, gt_exp)

            total_loss = loss_j3d + loss_j2d + loss_j2d_head + loss_j2d_hand \
                        + loss_pose_smplx + loss_shape_smplx + loss_exp_smplx
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
            self.writer.add_scalar('Loss/pose', loss_pose_smplx, global_step=self.global_step)
            self.writer.add_scalar('Loss/shape', loss_shape_smplx, global_step=self.global_step)
            self.writer.add_scalar('Loss/expression', loss_exp_smplx, global_step=self.global_step)
            self.writer.add_scalar('Loss/Total', total_loss, global_step=self.global_step)
            self.global_step+=1
    def eval_per_epoch(self, batch_generator, len_batch_loader):
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
            pred_cam = torch.stack([pred_dict['camera'][:,1],
                                pred_dict['camera'][:,2],
                                2* 5000./(256 * pred_dict['camera'][:,0] +1e-9)],dim=-1).detach().cpu().numpy()
           
            pass
        final_scores = PA_MPJPE_scores/len_batch_loader * 1000
        logger.debug(f'PA MPJPE scores in (mm) : {final_scores}')
        return final_scores
# class TrainVIBEX(object):
#     def __init__(self, conf, device ='cuda'):
#         self.conf = conf 
#         self.device = device
#         #--Setup human model requirements 
#         pose_desc_dict = load_all_mean_params(conf['all_mean_path'], conf['shape_mean_path'])
#         self.mean_params = get_param_mean(pose_desc_dict)
#         self.human_model = SMPLXLayer(pose_desc_dict, conf['human_model']['smplx']['model_path'])

#         #--Setup network, optimizer, and scheduler 
#         self.model = VIBEX(conf['seqlen'], self.mean_params.to(self.device), batch_size=conf['batch_size'])
#         self.model.to(self.device)
#         self.optim = Adam(lr = float(conf['lr']), params=self.model.parameters(), weight_decay= float(conf['lr_decay']))
#         self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, 
#                                             patience=conf['lr_patience'], verbose=True,)

#         #--Setup dataset
#         self.dset_3d =  Dataset3D(conf['datasets']['MPII3D'], conf['seqlen'], 
#                                     conf['pretrained_features'], )
#         self.loss_w = self.conf['loss_weights']
#         self.loss = HumanLoss()

#         #--Setup tensorboard for loss visualization
#         self.writer = SummaryWriter('logs')
#         self.global_step = 0
#     def __call__(self):
#         for epoch in range(self.conf['epochs']):
#             batch_loader = DataLoader(self.dset_3d, shuffle= True, 
#                             batch_size=self.conf['batch_size'],  num_workers= 8, 
#                             pin_memory= True, drop_last=True)
#             total_data = min(self.conf['max_inter_per_epoch'], len(batch_loader) )
#             batch_generator = tqdm(batch_loader, total = len(batch_loader))
#             self.train_per_epoch(batch_generator, total_data)
            
#             '''Save model'''
#             torch.save({
#                 'model': self.model.state_dict(),
#                 'model_opt': self.optim.state_dict()}, 
#                 '{}/temporal_VIBEX.h5'.format('data/weights/body'))
#     def train_per_epoch(self, batch_generator, len_batch_loader):
#         for inputs in batch_generator:
#             features = inputs['features'].to(self.device)
#             gt_j2d = inputs['kp_2d'].to(self.device)
#             gt_j3d = inputs['kp_3d'].to(self.device)
#             visible = gt_j2d[:, : , :, -1].unsqueeze(-1).clone()
                     
#             pred = self.model(features)
#             pred_smplx, raw_pred_param = self.human_model.get_smplx_from_video(pred)
#             pred_j3d = pred_smplx['joints']
#             pred_j2d = self.human_model.get_2d_pred(pred_j3d, raw_pred_param)
#             pred_j2d = pred_j2d[:, :, :2]

#             #--Reshape the prediction dimension to [batch size, seqlen, 144, :]
#             pred_j2d = pred_j2d.reshape(self.conf['batch_size'], self.conf['seqlen'], 
#                                         pred_j2d.shape[1], pred_j2d.shape[2])
#             pred_j3d = pred_j3d.reshape(self.conf['batch_size'], self.conf['seqlen'], 
#                                         pred_j3d.shape[1], pred_j3d.shape[2])
            
#             #--Compute loss
#             self.optim.zero_grad()
#             loss_2d = self.loss.compute_kp_2d(pred_j2d, gt_j2d[:, :, :, :2], visible) * self.loss_w['KP_2D_W'] 
#             loss_3d = self.loss.compute_kp_3d(pred_j3d, gt_j3d, visible) * self.loss_w['KP_3D_W']

#             total_loss = loss_2d + loss_3d
#             total_loss.backward()
            
#             #--Write in tensorboard
#             self.writer.add_scalar('Loss/3D_Keyps', loss_3d, global_step=self.global_step)
#             self.writer.add_scalar('Loss/2D_Keyps', loss_2d, global_step=self.global_step)
#             self.writer.add_scalar('Loss/Total', total_loss, global_step=self.global_step)
#             self.global_step+=1
#         draw_keypoints_w_black_bg(pred_j2d, gt_j2d[:, :, :, :2], crop_size=224, 
#                                         batch_size=self.conf['batch_size'], seqlen=self.conf['seqlen'])
#         # render_temporal(pred_smplx, raw_pred_param['camera'], self.human_model.faces)
#         self.scheduler.step(total_loss)