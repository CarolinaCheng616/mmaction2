import torch.nn as nn
import torch
import numpy as np
import os.path as osp

from .. import builder
from .base import FeatureExtractor
from ..registry import EXTRACTORS

# from ...utils import get_root_logger

# from mmcv.cnn import ConvModule, constant_init, kaiming_init
# from mmcv.runner import _load_checkpoint, load_checkpoint


@EXTRACTORS.register_module()
class VideoExtractor(FeatureExtractor):
    def __init__(self, backbone, train_cfg=None, test_cfg=None):
        super(VideoExtractor, self).__init__(backbone, train_cfg, test_cfg)
        self.backbone = builder.build_backbone(backbone)
        self.backbone.init_weights()
        self.backbone.eval()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # def init_weights(self):
    #     """Initiate the parameters either from existing checkpoint or from
    #     scratch."""
    #     if isinstance(self.pretrained, str):
    #         logger = get_root_logger()
    #         if self.torchvision_pretrain:
    #             self._load_torchvision_checkpoint(logger)
    #         else:
    #             load_checkpoint(
    #                 self, self.pretrained, strict=False, logger=logger)
    #     elif self.pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 constant_init(m, 1)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def forward_test(self, x, img_metas=None):
        # x.shape = BNCHW
        batch, frames = x.shape[:2]
        idxes = []
        new_img_metas = []
        for i in range(batch):
            feature_path = img_metas[i]["featurepath"]
            if not osp.exists(feature_path):
                idxes.append(i)
                new_img_metas.append(img_metas[i])
        idxes = np.array(idxes)
        if len(idxes) == 0:
            return
        x = x[idxes]
        img_metas = new_img_metas
        with torch.no_grad():
            batch, frames = x.shape[:2]
            x = x.reshape((-1,) + x.shape[2:])  # BN * CHW
            x = self.backbone(x)  # BN * 2048 * H' * W'
            x = self.avgpool(x)  # BN * 2048 * 1 * 1
            x = x.reshape(batch, frames, -1)  # B * N * 2048
            for i in range(batch):
                feature = x[i].unsqueeze(0).cpu().detach().numpy()  # 1, 32, 2048
                feature_path = img_metas[i]["featurepath"]
                np.save(feature_path, feature)

    def forward(self, imgs, return_loss=False, img_metas=None):
        self.forward_test(imgs, img_metas)
        return []
