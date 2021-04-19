import torch.nn as nn
import torch
import numpy as np
import sys

from .. import builder
from .base import FeatureExtractor


class VideoExtractor(FeatureExtractor):
    def __init__(self, backbone, train_cfg=None, test_cfg=None):
        super(VideoExtractor, self).__init__(backbone, train_cfg, test_cfg)
        self.backbone = builder.build_backbone(backbone)
        self.backbone.eval()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_test(self, x, video_meta=None):
        # x.shape = BNCHW
        with torch.no_grad():
            batch, frames = x.shape[:2]
            x = x.reshape((-1,) + x.shape[2:])  # BN * CHW
            x = self.backbone(x)  # BN * 2048 * H' * W'
            x = x.reshape(batch, frames, *x.shape[1:])  # B * N * 2048 * H' * W'
            x = self.avgpool(x)  # B * N * 2048 * 1 * 1
            x = x.reshape(batch, frames, -1)  # B * N * 2048
            for i in range(batch):
                feature = x[i].unsqueeze(0).cpu().detach().numpy()  # 1, 32, 2048
                feature_path = video_meta[i]["featurepath"]
                if feature_path.enswith(".npy"):
                    np.save(feature_path, feature)
        sys.exit(0)

    def forward(self, x, return_loss=False, video_meta=None):
        self.forward_test(x, video_meta)
        return 0