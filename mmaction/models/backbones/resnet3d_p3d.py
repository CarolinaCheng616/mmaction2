import math
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer, constant_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch import nn as nn
from torch.nn.modules.utils import _triple

from ...utils import get_root_logger
from ..registry import BACKBONES


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    pad_shape = out.shape
    pad_shape[1] = planes - pad_shape[1]
    zero_pads = torch.zeros(pad_shape, device=x.device)
    out = torch.cat([out, zero_pads], dim=1)

    return out


class BottleneckP3D(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 count_block,
                 layer_index,
                 spatial_stride=1,
                 temporal_stride=1,
                 downsample=None,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 p3d_style_seq=('A', 'B', 'C'),
                 with_cp=False):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.p3d_style = list(p3d_style_seq)[count_block % len(p3d_style_seq)]
        self.layer_index = layer_index
        self.count_block = count_block
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp

        if self.layer_index == 3:
            conv1_cfg = dict(type='Conv2d')
            norm1_cfg = dict(type='BN2d')
            conv1_stride = spatial_stride
        else:
            conv1_cfg = conv_cfg
            norm1_cfg = norm_cfg
            conv1_stride = (temporal_stride, spatial_stride, spatial_stride)
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=conv1_stride,
            bias=False,
            conv_cfg=conv1_cfg,
            norm_cfg=norm1_cfg,
            act_cfg=None)

        if self.layer_index == 3:
            self.conv2 = ConvModule(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=None)
            self.conv3 = ConvModule(
                planes,
                planes * 4,
                kernel_size=1,
                bias=False,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=None)
            self.conv4 = None
        else:
            self.conv2 = ConvModule(
                planes,
                planes,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
            self.conv3 = ConvModule(
                planes,
                planes,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
            self.conv4 = ConvModule(
                planes,
                planes * 4,
                kernel_size=1,
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.layer_index == 3 or self.p3d_style == 'A':
            self.inner_forward = self.a_forward
        elif self.p3d_style == 'B':
            self.inner_forward = self.b_forward
        elif self.p3d_style == 'C':
            self.inner_forward = self.c_forward

    def a_forward(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def b_forward(self, x):
        tmp_x = self.conv2(x)
        x = self.conv3(x)
        return x + tmp_x

    def c_forward(self, x):
        x = self.conv2(x)
        tmp_x = self.conv3(x)
        return x + tmp_x

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.inner_forward(out)
            if self.conv4 is not None:
                out = self.conv4(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNetP3D(nn.Module):

    arch_settings = {
        63: (BottleneckP3D, (3, 4, 6, 3)),
        131: (BottleneckP3D, (3, 4, 23, 3)),
        199: (BottleneckP3D, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=2,
                 with_pool2=True,
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False,
                 zero_init_residual=True,
                 shortcut_type='B',
                 p3d_style_seq=('A', 'B', 'C'),
                 **kwargs):
        super().__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_t = conv1_stride_t
        self.with_pool2 = with_pool2
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pool1_stride_t = pool1_stride_t
        self.shortcut_type = shortcut_type
        self.p3d_style_seq = p3d_style_seq
        self.zero_init_residual = zero_init_residual
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.count_block = 0

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                i,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                with_cp=with_cp,
                shortcut_type=self.shortcut_type,
                p3d_style_seq=p3d_style_seq,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(2, 3, 3),
            stride=(self.pool1_stride_t, 2, 2),
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       layer_index,
                       spatial_stride=1,
                       temporal_stride=1,
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None,
                       with_cp=False,
                       shortcut_type='B',
                       p3d_style_seq=('A', 'B', 'C'),
                       **kwargs):
        downsample = None

        if spatial_stride != 1 or inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=spatial_stride)
            else:
                if layer_index == 3:
                    downsample = ConvModule(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=spatial_stride,
                        bias=False,
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=dict(type='BN2d', requires_grad=True),
                        act_cfg=None)
                else:
                    downsample = ConvModule(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(temporal_stride, spatial_stride,
                                spatial_stride),
                        bias=False,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None)

        layers = list()
        layers.append(
            block(
                inplanes,
                planes,
                self.count_block,
                layer_index,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                downsample=downsample,
                p3d_style_seq=p3d_style_seq,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        self.count_block += 1

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    self.count_block,
                    layer_index,
                    p3d_style_seq=p3d_style_seq,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))
            self.count_block += 1

        return nn.Sequential(*layers)

    def _inflate_conv_params(self, conv3d, origin_dict, origin_name,
                             inflated_names):
        weight_2d_name = origin_name + '.weight'
        conv2d_weight = origin_dict[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = origin_name + '.bias'
            conv3d.bias.data.copy_(origin_dict[bias_2d_name])
            inflated_names.append(bias_2d_name)

    def _copy_conv_params(self, conv2d, origin_dict, origin_name, inflated_names):
        conv2d_weight = origin_dict[origin_name + '.weight']
        conv2d.weight.data.copy_(conv2d_weight.data)
        inflated_names.append(origin_name + '.weight')

        if getattr(conv2d, 'bias') is not None:
            conv2d.bias.data.copy_(origin_dict[origin_name + '.bias'])
            inflated_names.append(origin_name + '.bias')

    def _inflate_bn_params(self, bn3d, origin_dict, origin_name,
                           inflated_names):
        parameters = list(bn3d.named_parameters()) + list(bn3d.named_buffers())
        for name, param in parameters:
            bn2d_name = f'{origin_name}.{name}'
            if bn2d_name in origin_dict:
                param.data.copy_(origin_dict[bn2d_name])
                inflated_names.append(bn2d_name)

    def inflate_weights(self, logger):
        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'conv3' in name and 'layer4' not in name:
                    continue
                if 'conv4' in name and 'layer4' not in name:
                    name = name.replace('conv4', 'conv3')
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    shape_2d = state_dict_r2d[original_conv_name + '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:] and 'layer4' not in name:
                        logger.warning(f'Weight shape mismatch for '
                                       f': {original_conv_name} :'
                                       f'3d weight shape: {shape_3d};'
                                       f'2d weight shape: {shape_2d}. ')
                    elif 'layer4' in name:
                        self._copy_conv_params(module.conv,
                                               state_dict_r2d,
                                               original_conv_name,
                                               inflated_param_names)
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d,
                                                  original_conv_name,
                                                  inflated_param_names)

                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        remaining_names = set(state_dict_r2d.keys() -
                              set(inflated_param_names))
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                self.inflate_weights(logger)
            else:
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            # import pdb
            # pdb.set_trace()
            for m in self.modules():
                print(type(m))
                if isinstance(m, nn.Conv3d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BottleneckP3D):
                        if m.conv4 is not None:
                            constant_init(m.conv4.bn, 0)
                        else:
                            constant_init(m.conv3.bn, 0)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i != len(self.res_layers) - 1:
                x = res_layer(x)
                if self.with_pool2:
                    x = self.pool2(x)
            else:
                x = x.reshape(-1, x.shape[1], x.shape[3], x.shape[4])
                x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()


if __name__ == '__main__':
    # data = torch.rand(10, 3, 16, 160, 160, requires_grad=False)
    model = ResNetP3D(199)
    # out = model(data)
    model.init_weights()
