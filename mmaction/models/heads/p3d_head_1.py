import torch.nn as nn
import torch
from mmcv.cnn import normal_init
from mmcv.runner import load_state_dict

from ..registry import HEADS
from ...utils import get_root_logger
from .base import BaseHead


@HEADS.register_module()
class P3DHead1(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 pretrained=None,
                 dropout_ratio=0.0,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)
        self.init_std = init_std
        self.pretrained = pretrained
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5),
                                    stride=1)  # pooling layer for res5.
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Linear(in_channels, self.num_classes)

    def init_weights(self):
        if self.pretrained and isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'p3d_head_1 load model from: {self.pretrained}')
        else:
            normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        if self.avgpool is not None:
            x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)
        if self.dropout is not None:
            x = self.dropout(x)
        cls_scores = self.fc(x)

        return cls_scores


# if __name__ == '__main__':
#     import torch
#     import sys
#     model = P3DHead1(num_classes=400, in_channels=2048, pretrained=sys.argv[1])
#     model.init_weights()
#     # data = torch.rand(10, 2048, 5, 5)
#     # out = model(data)
