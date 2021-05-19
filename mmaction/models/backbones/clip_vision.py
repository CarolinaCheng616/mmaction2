import torch.nn as nn

from ...utils import get_root_logger
from ..registry import BACKBONES

import clip


@BACKBONES.register_module()
class CLIP(nn.Module):
    def __init__(self, pretrained=None, freeze=True):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            assert (
                self.pretrained in clip.available_models()
            ), "not allowed pretrained model"
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")
            self.model = clip.load(self.pretrained)[0]
        else:
            raise TypeError("pretrained must be a str")

    def forward(self, x):
        # x.shape = [N, C, H, W]
        import pdb

        pdb.set_trace()
        if self.freeze:
            self.model.eval()
        features = self.model.encode_image(x)
        return features
