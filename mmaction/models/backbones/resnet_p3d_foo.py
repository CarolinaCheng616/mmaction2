import torch.nn as nn
import torch

from mmcv.cnn import ConvModule

from .resnet import Bottleneck
from .resnet import ResNet
from ..registry import BACKBONES


class BottleneckP3D(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47, st_struct=('A', 'B', 'C')):
        super().__init__(
            inplanes, planes, downsample=downsample,
        )
        self.downsample = downsample

        self.depth_3d = depth_3d
        self.st_struct = st_struct
        self.len_st = len(self.st_struct)
        self.id = n_s
        self.st = list(self.st_struct)[self.id % self.len_st]
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        conv3d = dict(type='Conv3d')
        conv2d = dict(type='Conv2d')
        bn3d = dict(type='BN3d')
        bn2d = dict(type='BN2d')
        relu = dict(type='ReLU', inplace=True)

        stride_p = stride
        if self.downsample is not None:
            stride_p = (1, 2, 2)
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.block1 = ConvModule(inplanes, planes, kernel_size=1, bias=False, stride=stride_p,
                                     conv_cfg=conv3d, norm_cfg=bn3d, act_cfg=relu)
            self.block_s = ConvModule(inplanes, planes, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                      bias=False, conv_cfg=conv3d, norm_cfg=bn3d, act_cfg=relu)
            self.block_t = ConvModule(inplanes, planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                                      bias=False, conv_cfg=conv3d, norm_cfg=bn3d, act_cfg=relu)
            self.block4 = ConvModule(planes, planes * 4, kernel_size=1, bias=False,
                                     conv_cfg=conv3d, norm_cfg=bn3d, act_cfg=None)
        else:
            stride_p = 2 if n_s == self.depth_3d else 1
            self.block1 = ConvModule(inplanes, planes, kernel_size=1, bias=False, stride=stride_p,
                                     conv_cfg=conv2d, norm_cfg=bn2d, act_cfg=relu)
            self.block_normal = ConvModule(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                           conv_cfg=conv2d, norm_cfg=bn2d, act_cfg=relu)
            self.block4 = ConvModule(planes, planes * 4, kernel_size=1, bias=False,
                                     conv_cfg=conv2d, norm_cfg=bn2d, act_cfg=None)

    def struct_a(self, x):
        s = self.block_s(x)
        ts = self.block_t(s)
        return ts

    def struct_b(self, x):
        s = self.block_s(x)
        t = self.block_t(x)
        return s + t

    def struct_c(self, x):
        s = self.block_s(x)
        ts = self.block_t(s)
        return s + ts

    def forward(self, x):
        residual = x

        out = self.block1(x)

        if self.id < self.depth_3d:
            if self.st == 'A':
                out = self.struct_a(out)
            elif self.st == 'B':
                out = self.struct_b(out)
            elif self.st == 'C':
                out = self.struct_c(out)
        else:
            out = self.block_normal(out)

        out = self.block4(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNetP3D(ResNet):

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, modality='RGB', shortcut_type='B',
                 num_classes=400, dropout=0.5, st_struc=('A', 'B', 'C')):
        structure = self.arch_settings[depth]
        block, layers = structure[0], structure[1]
        self.in_channels = 3 if modality == 'RGB' else 2
        super().__init__(
            depth,
        )
        pass

    def forward(self, x):
        pass
