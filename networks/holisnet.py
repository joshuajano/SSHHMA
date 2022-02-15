from turtle import forward
import torch
import torch.nn as nn
import os 
from loguru import logger

from networks.hrnet import PoseHighResolutionNet
from networks.resnet import PoseResNet
from networks.blocks import BasicBlock, BN_MOMENTUM
class HolisNet(nn.Module):
    def __init__(self, conf, mean_params, layers = [3, 4, 6, 3]):
        super().__init__()
        self.inplanes = 64
        self.conf = conf
        self.backbone = PoseHighResolutionNet(conf['networks']['hrnet'])
        
        self.backbone.init_weights(conf['networks']['hrnet']['weight_path'])
        #--Final SMPL-X parameters
        self.mean_params = mean_params
        self.fc1 = nn.Linear(512 * 4 + mean_params.shape[1], 1024)
        
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
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
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decexp.weight, gain=0.01)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x, n_iter=3):
        batch_size = x.shape[0]
        _init = self.mean_params.expand(batch_size, -1)
        
        pred_pose = _init[:, :318]
        pred_shape = _init[:, 318: 328]
        pred_exp = _init[:, 328: 338]
        pred_cam = _init[:, 338: 341]

        features = self.backbone(x)

        #--Use residual extraction
        x1 = self.layer1(features)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_exp, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_exp = self.decexp(xc) + pred_exp
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_param = torch.cat([pred_pose, pred_shape, pred_exp, pred_cam], 1)
        return pred_param
        
