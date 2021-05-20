import torch.nn as nn
import torch

from ...utils import get_root_logger
from ..registry import BACKBONES

import clip


@BACKBONES.register_module()
class CLIP(nn.Module):
    def __init__(self, pretrained=None, freeze=True, fp16_enabled=True):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.fp16_enabled = fp16_enabled

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            assert (
                self.pretrained in clip.available_models()
            ), "not allowed pretrained model"
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")
            self.model = clip.load(self.pretrained)[0].visual
        else:
            raise TypeError("pretrained must be a str")

    def forward(self, x):
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
