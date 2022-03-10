import os
from loguru import logger 
import torch
import pickle
import gc
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torchgeometry as tgm
# from datasets.dataset_3d import Dataset3D
from networks.holisnet import HolisNet
from datasets.threedpw_eval import ThreedpwEval
from datasets.h36m import H36M
from human_models.smplxLayer import SMPLXLayer
from commons.smplx_utils import load_all_mean_params, get_param_mean
from losses.loss import HumanLoss
from commons.render import Renderer
from commons.human_mesh_loader import HumanMeshLoader
from human_models.smplxLayer import SMPLXLayer
from commons.eval_utils import reconstruction_error
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
class TestAnnotator(object):
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
        logger.success('success load the pretrained {} weights'.format(conf['networks']['checkpoint_weight']))
        self.net.to(device)
        
        #--Prepare dataset
        self.threedpw_eval_dset = ThreedpwEval(conf['datasets'])
        #--Setup tensorboard for loss visualization
        self.global_step = 0
        
        #--Setup for evaluation 
        with open('data/SMPLX_to_j14.pkl', 'rb') as f:
            self.J14_regressor = pickle.load(f, encoding='latin1')
        self.SMPLX_reg_to_LSP = torch.from_numpy(self.J14_regressor).float()
        self.render = Renderer(faces = self.pred_human_model.faces)
        self.save_train_dir = 'visualize/test/h36m'
        if not os.path.exists(self.save_train_dir):
            os.makedirs(self.save_train_dir)
    def eval_3DPW(self):
        logger.debug('Evaluation mode on 3DPW')
        eval_batch_loader = DataLoader(self.threedpw_eval_dset, shuffle= False, 
                                batch_size=1,  num_workers= 2, 
                                pin_memory= True, drop_last=True)
        eval_batch_generator = tqdm(eval_batch_loader, total = len(eval_batch_loader))
        PA_MPJPE_scores = 0
        for inputs in eval_batch_generator:
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
        final_scores = PA_MPJPE_scores/len(eval_batch_loader) * 1000
        logger.debug(f'PA MPJPE scores in (mm) : {final_scores}')
        return final_scores
    def test_on_h36m(self, out_path='/home/josh/Desktop/Inha/My/trial/h36m/npz/'):
        _imgnames, _scales, _centers, _parts, _Ss = [], [], [], [], []
        _poses, _shapes, _exps = [], [], []
        self.net.eval()
        test_dset = H36M(self.conf['datasets'])
        batch_loader = DataLoader(test_dset, shuffle= False, 
                            batch_size=1, num_workers= 1, 
                            pin_memory= True, drop_last=False)
        train_batch_generator = tqdm(batch_loader, total = len(batch_loader))
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        for inputs in train_batch_generator:
            img = inputs['img'].to(self.device)
            im_name = inputs['img_fn']
            kp2d = inputs['kp2d']
            kp3d = inputs['kp3d']
            scale = inputs['scale']
            center = inputs['center']

            with torch.no_grad():
                pred_param = self.net(img)
                #--Convert from 6D to axis-angle
                pred_dict = self.human_model.flat_body_params_to_dict(pred_param)
                pred_rot_mat = self.human_model.convert_6D_to_rot_mat(pred_dict)
                pred_mesh = self.pred_human_model.gen_smplx(pred_rot_mat)
            
            self.global_step+=1
            # logger.success(im_name)
            #--Tensor to numpy 
            kp2d = kp2d[0].detach().cpu().numpy()
            kp3d = kp3d[0].detach().cpu().numpy()
            scale = scale[0].detach().cpu().numpy()
            center = center[0].detach().cpu().numpy()
            im_name = im_name[0]
            shape = pred_mesh.betas[0].detach().cpu().numpy()
            exp = pred_mesh.expression[0].detach().cpu().numpy()
            #--Convert rotmat to axis angle
            all_poses = torch.cat([pred_mesh.global_orient[0], pred_mesh.body_pose[0], 
                                    pred_mesh.left_hand_pose[0], pred_mesh.right_hand_pose[0], pred_mesh.jaw_pose[0]]
                                    , dim= 0)
            all_poses = pose = torch.cat([all_poses, torch.zeros((all_poses.shape[0],3,1)).cuda().float()],2)
            all_poses = tgm.rotation_matrix_to_angle_axis(all_poses)
            pose = all_poses.detach().cpu().numpy()
            #--Update the dataset
            _imgnames.append(im_name)
            _scales.append(scale)
            _centers.append(center)
            _parts.append(kp2d)
            _Ss.append(kp3d)
            _poses.append(pose)
            _shapes.append(shape)
            _exps.append(exp)
            if self.global_step>0 and self.global_step%500000==0:
                out_file = os.path.join(out_path, 'h36m_train_w_pseudo_SMPLX_35_{}.npz'.format(self.global_step))
                np.savez(out_file, imgname=_imgnames,
                           center=_centers,
                           scale=_scales,
                           part=_parts,
                           S=_Ss,
                           pose = _poses,
                           shape = _shapes,
                           expression = _exps
                           )
                logger.success(f'saved h36m with  pseudo {out_file}')
                del _imgnames, _scales, _centers, _parts, _Ss
                del _poses, _shapes, _exps
                gc.collect()
                _imgnames, _scales, _centers, _parts, _Ss = [], [], [], [], []
                _poses, _shapes, _exps = [], [], []
        out_file = os.path.join(out_path, 'h36m_train_w_pseudo_SMPLX_20_{}.npz'.format(self.global_step))
        np.savez(out_file, imgname=_imgnames,
                   center=_centers,
                   scale=_scales,
                   part=_parts,
                   S=_Ss,
                   pose = _poses,
                   shape = _shapes,
                   expression = _exps
                   )
        logger.success(f'saved h36m with  pseudo {out_file}')
            
#--Render
# if self.global_step%1500 ==0:
#     pred_cam = torch.stack([pred_dict['camera'][:,1],
#                 pred_dict['camera'][:,2],
#                 2* 5000./(256 * pred_dict['camera'][:,0] +1e-9)],dim=-1).detach().cpu().numpy()
#     images = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
#     images = images + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
#     img_np = images[0].permute((1, 2, 0)).detach().cpu().numpy() 
#     vert = pred_mesh.vertices[0].detach().cpu().numpy()
#     rend_img = self.render(vert, pred_cam[0], img_np)
#     rend_img *= 255.
#     save_name = os.path.join(self.save_train_dir, f'{self.global_step}.jpg')
#     cv2.imwrite(save_name, cv2.cvtColor(rend_img.astype(np.uint8), cv2.COLOR_BGR2RGB ))