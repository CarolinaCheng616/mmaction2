import torch.nn as nn
import torch
import torch.distributed as dist
import numpy as np

from ..backbones import bert
from ..registry import EXTRACTORS


@EXTRACTORS.register_module()
class BertExtractor(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
