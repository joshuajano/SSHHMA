import os 
from loguru import logger
import smplx
import torch

class HumanMeshLoader(object):
    def __init__(self, model_folder, pose_desc_dict, model_type='smplx', num_betas = 10) -> None:
        super().__init__()
        #--Load all genders and kids model
        #--Initialize get kid template
        logger.warning(f'Initializing {model_type} human model')
        if model_type =='smplx':
            KID_TEMPLATE = 'smplx_kid_template.npy'
        else:
            KID_TEMPLATE = 'smpl_kid_template.npy'
        KID_PATH = os.path.join(model_folder, 'kids', KID_TEMPLATE)
        self.model_male_adult = smplx.create(model_folder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=num_betas, use_pca=False)
        logger.success(f'succesfully load male adult with {num_betas} number of shape')     
        self.model_female_adult = smplx.create(model_folder, model_type='smplx',
                                  gender='female',
                                  ext='npz',
                                  num_betas=num_betas, use_pca=False)
        logger.success(f'succesfully load female adult with {num_betas} number of shape')    
        self.model_neutral_adult = smplx.create(model_folder, model_type='smplx',
                                  gender='neutral',
                                  ext='npz',
                                  num_betas=num_betas, use_pca=False)
        logger.success(f'succesfully load neutral adult with {num_betas} number of shape')
        self.pose_desc_dict = pose_desc_dict
        self.get_idxs(pose_desc_dict)
    def get_idxs(self, pose_desc_dict, 
                    num_betas = 10, 
                    num_expression_coeffs = 10, 
                    cam_param_num = 3, dtype = torch.long):
        start = 0 
        #get global orient idxs
        global_orient_dim = pose_desc_dict['global_orient']['dim']
        self.global_orient_idxs = torch.tensor(list(range(start, start + global_orient_dim)), dtype= dtype).cuda()
        start += global_orient_dim
        #get body pose idxs
        body_pose_dim = pose_desc_dict['body_pose']['dim']
        self.body_pose_idxs = torch.tensor(list(range(start, start + body_pose_dim)), dtype= dtype).cuda()
        start += body_pose_dim
        #get left hand pose idxs
        left_hand_pose_dim = pose_desc_dict['left_hand_pose']['dim']
        self.left_hand_pose_idxs = torch.tensor(list(range(start, start + left_hand_pose_dim)), dtype= dtype).cuda()
        start += left_hand_pose_dim
        #get left hand pose idxs
        right_hand_pose_dim = pose_desc_dict['right_hand_pose']['dim']
        self.right_hand_pose_idxs = torch.tensor(list(range(start, start + right_hand_pose_dim)), dtype= dtype).cuda()
        start += right_hand_pose_dim
        #get jaw pose idxs
        jaw_pose_dim = pose_desc_dict['jaw_pose']['dim']
        self.jaw_pose_idxs = torch.tensor(list(range(start, start + jaw_pose_dim)), dtype= dtype).cuda()
        start += jaw_pose_dim
        #additional for shape, expression, camera parameter
        #shape
        shape_dim = num_betas
        self.shape_idxs = torch.tensor(list(range(start, start + shape_dim)), dtype= dtype).cuda()
        start += shape_dim
        #expression
        exp_dim = num_expression_coeffs
        self.expression_idxs = torch.tensor(list(range(start, start + exp_dim)), dtype= dtype).cuda()
        start += exp_dim
        #camera 
        cam_dim = cam_param_num
        self.camera_idxs = torch.tensor(list(range(start, start + cam_dim)), dtype= dtype).cuda()
        start += cam_dim
    def flat_body_params_to_dict(self, param_tensor):
        '''EDITED change using cuda variable'''
        global_orient = torch.index_select(
            param_tensor, 1, self.global_orient_idxs)
        body_pose = torch.index_select(
            param_tensor, 1, self.body_pose_idxs)
        left_hand_pose = torch.index_select(
            param_tensor, 1, self.left_hand_pose_idxs)
        right_hand_pose = torch.index_select(
            param_tensor, 1, self.right_hand_pose_idxs)
        jaw_pose = torch.index_select(
            param_tensor, 1, self.jaw_pose_idxs)
        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)
        camera = torch.index_select(param_tensor, 1, self.camera_idxs )
        return {
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'camera': camera
        } 
    def convert_6D_to_rot_mat(self, param):
        global_orient = self.pose_desc_dict['global_orient']['decoder'](param['global_orient'].clone())
        body_pose = self.pose_desc_dict['body_pose']['decoder'](param['body_pose'])
        left_hand_pose = self.pose_desc_dict['left_hand_pose']['decoder'](param['left_hand_pose'])
        right_hand_pose = self.pose_desc_dict['right_hand_pose']['decoder'](param['right_hand_pose'])
        jaw_pose = self.pose_desc_dict['jaw_pose']['decoder'](param['jaw_pose'])

        full_pose = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose], dim=1)
        return {'betas': param['betas'],
                'expression': param['expression'],
                'global_orient': global_orient,
                'body_pose': body_pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                'jaw_pose': jaw_pose,
                'camera': param['camera'],
                'full_pose': full_pose.view(-1, 3, 3)
        }
        # self.model_male_kid = smplx.create(model_folder, model_type=model_type,
        #                               gender='male',
        #                               age='kid',
        #                               kid_template_path=KID_PATH,
        #                               ext='npz', use_pca=False)
        # logger.success(f'succesfully load male kid')
        # self.model_female_kid = smplx.create(model_folder, model_type=model_type,
        #                               gender='neutral',
        #                               age='kid',
        #                               kid_template_path=KID_PATH,
        #                               ext='npz', use_pca=False)
        # logger.success(f'succesfully load female kid')
        # self.model_neural_kid = smplx.create(model_folder, model_type=model_type,
        #                               gender='neutral',
        #                               age='kid',
        #                               kid_template_path=KID_PATH,
        #                               ext='npz', use_pca=False)
        # logger.success(f'succesfully load neutral kid')