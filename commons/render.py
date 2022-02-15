import os
import numpy as np
import pyrender
import trimesh
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from torchvision.utils import make_grid

import cv2
import math

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()
def render_temporal(param, cam_param, faces, width= 224, height= 224, 
                    batch_size=32, seqlen=16, 
                    fl = 5000., 
                    base_color = (0.8, 0.3, 0.3, 1.0), save_dir ='visualize/train'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    black_bg = np.zeros((batch_size, seqlen, width, height, 3))
    #--Setup camera parameter 
    cam_param = cam_param.reshape(batch_size, seqlen, cam_param.shape[1])
    meshes = param.vertices
    meshes = meshes.reshape(batch_size, seqlen, meshes.shape[1], meshes.shape[2])
    for i in range(2):
        for j in range(seqlen):
            pred_cam = torch.stack([cam_param[i, j, 1],
                            cam_param[i, j, 2],
                            2*fl/(width * cam_param[i, j, 0] + 1e-9)], dim=-1)
            rend_img = do_render(black_bg[i, j], meshes[i, j].detach().cpu().numpy(), 
                                faces, pred_cam.detach().cpu().numpy(), 
                                width=width, height=height) 
            rend_img *= 255
            savename_path = os.path.join(save_dir, f'{i}_{j}.jpg')
            save_obj(meshes[i, j].detach().cpu().numpy(), faces) 
            savename_path = os.path.join(save_dir, f'{i}_{j}.jpg')
            cv2.imwrite(savename_path, rend_img.astype(np.uint8))
def do_render(image, vertices, faces, 
                camera_translation, width= 224, height= 224,
                fl = 5000.,  
                base_color = (0.8, 0.3, 0.3, 1.0)):
    renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                       viewport_height=height,
                                       point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor= base_color)

    camera_translation[0] *= -1.
    mesh = trimesh.Trimesh(vertices, faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh, 'mesh')
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    
    camera = pyrender.IntrinsicsCamera(fx=fl, fy=fl,
                                       cx= width//2 , cy= height//2 )
    
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)
    
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:,:,None]
    output_img = (color[:, :, :3] * valid_mask +
              (1 - valid_mask) * image)
    
    return output_img
    
class Visualization_render(object):
    def __init__(self, focal_length, img_size, faces, root_dir ='visualize'):
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.deb_dir = os.path.join(root_dir, 'deblur')
        if not os.path.exists(self.deb_dir):
            os.makedirs(self.deb_dir)
        self.overlay_dir = os.path.join(root_dir, 'render')
        if not os.path.exists(self.overlay_dir):
            os.makedirs(self.overlay_dir)
        self.eval_3dpw_dir = os.path.join(root_dir, '3DPW')
        if not os.path.exists(self.eval_3dpw_dir):
            os.makedirs(self.eval_3dpw_dir)
        self.eval_SINBlur_dir = os.path.join(root_dir, 'SINBlur')
        if not os.path.exists(self.eval_SINBlur_dir):
            os.makedirs(self.eval_SINBlur_dir)
        self.eval_EHF_dir = os.path.join(root_dir, 'EHF')
        if not os.path.exists(self.eval_EHF_dir):
            os.makedirs(self.eval_EHF_dir)
        
        self.renderer = Renderer(focal_length=focal_length, img_res=img_size, faces=faces)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    # def __init__(self, focal_length=5000, img_res=224, faces=None):
    #     self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
    #                                    viewport_height=img_res,
    #                                    point_size=1.0)
    #     self.focal_length = focal_length
    #     self.camera_center = [img_res // 2, img_res // 2]
    #     self.faces = faces

    def __init__(self, focal_length=5000, img_res=256, width = 256, height = 256, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                       viewport_height=height,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [width // 2, height // 2]
        self.faces = faces
        self.loop_number = 0
    def __call__(self, vertices, camera_translation, image, cam_center = None, cam_for_render = None, base_color = (0.8, 0.3, 0.3, 1.0)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor= base_color)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        if cam_center is None:
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                               cx=self.camera_center[0], cy=self.camera_center[1])
        else:
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                               cx=cam_center[0], cy=cam_center[1])
        if cam_for_render is not None:
            camera = pyrender.IntrinsicsCamera(fx=cam_for_render[0], fy=cam_for_render[0],
                                           cx=cam_for_render[1], cy=cam_for_render[2])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        
        if cam_for_render is not None:
            output_img = (255 * color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
    