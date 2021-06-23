import torch.nn as nn

from ...utils import get_root_logger
from ..registry import BACKBONES

from timesformer.models.vit import TimeSformer


@BACKBONES.register_module()
class Timesformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_classes=240,
        num_frames=8,
        attention_type="divided_space_time",
        pretrained=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.attention_type = attention_type
        self.pretrained = pretrained
        assert (
            self.pretrained is not None
        ), "pretrained model path must be str, but not None."
        logger = get_root_logger()
        logger.info(f"load model from: {self.pretrained}")
        self.model = TimeSformer(
            img_size=img_size,
            num_classes=num_classes,
            num_frames=num_frames,
            attention_type=attention_type,
            pretrained_model=pretrained,
        )

    def init_weights(self):
        pass

    def forward(self, x):
        feature = self.model.model.forward_features(x)
        return feature
