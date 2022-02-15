import os
import yaml
import torch
from loguru import logger
from commons.smplx_utils import load_all_mean_params, get_param_mean
from commons.human_mesh_loader import HumanMeshLoader
from commons.render import Renderer, save_obj
from commons.demo_utils import get_demo_input
from human_models.smplxLayer import SMPLXLayer
from networks.holisnet import HolisNet

#--Setuep the device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device ='cpu'
#--Read all configuration
yaml_file = open('data/conf.yaml') 
conf = yaml.load(yaml_file, Loader=yaml.FullLoader)

#--Get template from SMPL-X original implementation
pose_desc_dict = load_all_mean_params(conf['all_mean_path'], conf['shape_mean_path'])
mean_params = get_param_mean(pose_desc_dict)
human_model = HumanMeshLoader(conf['smplx']['model_path'], pose_desc_dict)
pred_human_model = SMPLXLayer(pose_desc_dict, conf['smplx']['model_path'])

#--Load the network and its pretrained weight 
net = HolisNet(conf, mean_params.to(device))
checkpoint = torch.load(conf['networks']['checkpoint_weight'])
net.load_state_dict(checkpoint['model'])

#--Setup the renderer
# render = Renderer(faces = pred_human_model.faces)
save_demo_dir = 'output_demo'
if not os.path.exists(save_demo_dir):
    os.makedirs(save_demo_dir)

img = get_demo_input('examples/test.jpg')
with torch.no_grad():
    pred_param = net(img)
    #--Convert from 6D to axis-angle
    pred_dict = human_model.flat_body_params_to_dict(pred_param)
    pred_rot_mat = human_model.convert_6D_to_rot_mat(pred_dict)
    pred_mesh = pred_human_model.gen_smplx(pred_rot_mat)

#--Save to .obj file
pred_vertices = pred_mesh.vertices[0].detach().cpu().numpy()
save_path = os.path.join(save_demo_dir, 'output.obj')
save_obj(pred_vertices, pred_human_model.faces, save_path)




