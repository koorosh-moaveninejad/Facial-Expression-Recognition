# -*- coding: utf-8 -*-

import torch.nn as nn
from resnet import *
from utils import *

#Flatten 2D feature maps into 1D vector, used in uncertainty branches
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class res18feature(nn.Module):
    def __init__(self, args, pretrained=True, num_classes=7, drop_rate=0.4, out_dim=64):
        super(res18feature, self).__init__()

        #Create ResNet-18 from scratch
        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Load pretrained weights from renet18.msceleb.pth (trained on Ms-Celeb-1M face recognition dataset)
        msceleb_model = torch.load(args.pretrained_backbone_path, map_location=device)
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.out_dim = out_dim
        #Takes all layers of ResNet-18 except the last two (Global Average Pooling + Final FC layer).
        self.features = nn.Sequential(*list(res18.children())[:-2])

        #Uncertainty Branches
        #Learn the main feature representation ( mean)
        self.mu = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

        #Learn the uncertainty
        self.log_var = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

    def forward(self, x, target, phase='train'):

        if phase == 'train':
            x = self.features(x)
            mu = self.mu(x)
            logvar = self.log_var(x)

            # === Relative Uncertainty Mixup ===
            mixed_x, y_a, y_b, att1, att2 = mixup_data(mu, target, logvar.exp().mean(dim=1, keepdim=True), use_cuda=True)
            return mixed_x, y_a, y_b, att1, att2
        else:
            x = self.features(x)
            output = self.mu(x)
            return output
