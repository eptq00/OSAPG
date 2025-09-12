# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import copy
import warnings

import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.ops import DeformConv2d, batched_nms
from mmcv.runner import BaseModule, ModuleList



class AdaptiveConv(BaseModule):
    """AdaptiveConv used to adapt the sampling location with the anchors.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 3
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: False.
        type (str, optional): Type of adaptive conv, can be either 'offset'
            (arbitrary anchors) or 'dilation' (uniform anchor).
            Default: 'dilation'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 bias=False,
                 type='dilation',
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv'))):
        super(AdaptiveConv, self).__init__(init_cfg)
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1 and groups == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x, offset):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x