import torch.nn as nn

from .. import builder

from abc import ABCMeta, abstractmethod


class FeatureExtractor(nn.Module, metaclass=ABCMeta):
    def __init__(self, backbone, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def forward_test(self, imgs, texts, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""

    def forward(self, data):
        """Define the computation performed at every call."""
        pass
