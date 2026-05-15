# -*- coding: utf-8 -*-

import torch.nn as nn
from resnet import *
from utils2 import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class res18feature(nn.Module):
    """
    RUL model with experiment support.

    Controlled by args.experiment:
      'baseline'            — original RUL, random pairs, mean uncertainty, fixed 0.5/0.5 loss
      'label_smoothing'     — same as baseline but loss uses label smoothing (set in main.py)
      'dynamic_loss'        — uncertainty-weighted loss instead of fixed 0.5/0.5
      'entropy_uncertainty' — entropy-based uncertainty instead of mean variance
      'hard_negative'       — hard negative pairing instead of random pairing
    """

    def __init__(self, args, pretrained=True, num_classes=8, drop_rate=0.4, out_dim=64):
        super(res18feature, self).__init__()

        self.experiment = getattr(args, 'experiment', 'baseline')

        # Build ResNet-18 backbone and load pretrained weights
        res18 = ResNet(
            block=BasicBlock,
            n_blocks=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            output_dim=1000
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        msceleb_model = torch.load(args.pretrained_backbone_path, map_location=device)
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.out_dim = out_dim

        # Shared feature extractor — all layers except AvgPool and FC
        self.features = nn.Sequential(*list(res18.children())[:-2])

        # Branch A: facial expression embedding (mu)
        self.mu = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5)
        )

        # Branch B: uncertainty (log variance)
        self.log_var = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5)
        )

    def forward(self, x, target, phase='train'):

        if phase == 'train':
            x = self.features(x)
            mu = self.mu(x)
            logvar = self.log_var(x)

            # ── Uncertainty computation ──────────────────────────────────────
            if self.experiment == 'entropy_uncertainty':
                # EXPERIMENT: entropy of softmax(logvar) across embedding dims
                # Captures spread of uncertainty, not just average magnitude
                uncertainty = compute_uncertainty_entropy(logvar)
            else:
                # BASELINE: mean of exp(logvar) — original RUL
                uncertainty = compute_uncertainty_mean(logvar)

            # ── Pairing strategy ─────────────────────────────────────────────
            if self.experiment == 'hard_negative':
                # EXPERIMENT: pair with most similar sample from different class
                mixed_x, y_a, y_b, att1, att2 = mixup_data_hard_negative(
                    mu, target, uncertainty, use_cuda=True
                )
            else:
                # BASELINE: random pairing
                mixed_x, y_a, y_b, att1, att2 = mixup_data(
                    mu, target, uncertainty, use_cuda=True
                )

            return mixed_x, y_a, y_b, att1, att2

        else:
            # Inference: just return the mu embedding, no mixup
            x = self.features(x)
            output = self.mu(x)
            return output
