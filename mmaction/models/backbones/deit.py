import torch.nn as nn
import torch

from ...utils import get_root_logger
from ..registry import BACKBONES

import timm


@BACKBONES.register_module()
class DEIT(nn.Module):
    def __init__(self, pretrained=None, freeze=True, fp16_enabled=True):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.fp16_enabled = fp16_enabled

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")
            self.model = torch.hub.load(
                "facebookresearch/deit:main", self.pretrained, pretrained=True
            )
        else:
            raise TypeError("pretrained must be a str")

    def forward(self, x):
        import pdb

        pdb.set_trace()
        # x.shape = [batch * seg, C, H, W]
        if self.fp16_enabled:
            x = x.half()
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)
        # x.shape = [batch * seg, 512]
        return features
