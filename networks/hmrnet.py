import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from loguru import logger
from collections import OrderedDict
#from utils.geometry import rot6d_to_rotmat
""" We utilize HMR backbone , 
Human mesh recovery , CVPR 2018
"""
class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    # def __init__(self, block, layers, smpl_mean_params):
    def __init__(self, block, layers, mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.fc1 = nn.Linear(8533, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()
        self.final = nn.Linear(1024, 341)
        # self.decpose = nn.Linear(1024, npose)
        # self.decshape = nn.Linear(1024, 10)
        # self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.final.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # mean_params = mean_params
        # mean_params = np.load(smpl_mean_params)
        # init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        # init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        # init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('mean_param', mean_params)
        # self.register_buffer('init_pose', init_pose)
        # self.register_buffer('init_shape', init_shape)
        # self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, struct_feat, combineLayer, is_struct =False, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        _init = self.mean_param.expand(batch_size, -1)
        # if init_pose is None:
        #     init_pose = self.init_pose.expand(batch_size, -1)
        # if init_shape is None:
        #     init_shape = self.init_shape.expand(batch_size, -1)
        # if init_cam is None:
        #     init_cam = self.init_cam.expand(batch_size, -1)
        #We change first layer only

        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #change blur data with struture image feature
        #===Add===
        if is_struct:
            img_cat_struct = torch.cat((x.clone(), struct_feat), 1)
            addt_feat = combineLayer(img_cat_struct)
            x = addt_feat
        # img_cat_struct = torch.cat((x[blur_idx].clone(), struct_feat), 1)
        # addt_feat = combineLayer(img_cat_struct)
        # x[blur_idx] = addt_feat
        # x[blur_idx] = struct_feat

        #=========
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        a = []
        pred_param = _init
        for i in range(n_iter):
            xc = torch.cat([xf, pred_param],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_param = self.final(xc) + pred_param
            a.append(pred_param)
        pred_param = pred_param
        return pred_param
  
def hmr(smpl_mean_params):
    model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params)
    logger.debug('Load Body network')
    return model
