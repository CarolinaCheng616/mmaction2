from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .i3d_head import I3DHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tpn_head import TPNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .p3d_head import P3DHead
from .p3d_head_1 import P3DHead1

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'P3DHead', 'P3DHead1'
]
