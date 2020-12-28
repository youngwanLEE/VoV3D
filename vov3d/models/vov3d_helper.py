#!/usr/bin/env python3
# Copyright Youngwan Lee (ETRI). All Rights Reserved.

import torch
import torch.nn as nn

from vov3d.models.operators import Swish
from vov3d.models.operators import SE


class T_OSA_2plus1_Transform(nn.Module):
    """
    Temporal One-Shot Aggregation transformation with D(2+1)D
    : [Tx3x3] x num_groups where T is the size of
        temporal kernel and num_groups is the number of D(2+1)D.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups=4,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups  (int): number of conv in OSA module.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation along with temporal-axis.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(T_OSA_2plus1_Transform, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        norm_module,
    ):
        # keeps temporal dimension
        self.branch2a = False

        self.stride = stride
        dim_plane = dim_in

        self.layers = nn.ModuleList()
        # temp_stride = 2
        for idx in range(num_groups):
            self.layers.append(
                D_2plus1_D_bottlenck(
                    dim_plane, dim_inner, stride, idx, norm_module
                )
            )
            stride = 1
            dim_plane = int(dim_inner / 2)

        dim_inner = dim_plane * num_groups
        if self.stride == 1:
            dim_inner = dim_plane * num_groups + dim_in

        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):

        output = []
        # stride == 2: input feature not added.
        if self.stride == 1:
            output = [x]

        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)  # (N, C, T, H, W)

        # aggregation
        x = self.c(x)
        x = self.c_bn(x)

        return x


class D_2plus1_D_bottlenck(nn.Module):
    """
    Depthwise factorized component, D(2+1)D
    1x1x1 conv with dim_inner channel
    1x3x3 depthwise spatial conv with dim_inner channel
    3x1x1 depthwise temporal conv with dim_inner channel
    1x1x1 conv with (dim_inner/2) channel
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        stride=1,
        block_idx=0,
        norm_module=nn.BatchNorm3d,
        eps=1e-5,
        bn_mmt=0.1,
        se_ratio=0.0625,
        swish_inner=True,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_inner (int): the inner dimension of the block.
            stride (int): the stride of the bottleneck.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
                channel dimensionality being se_ratio times the Tx3x3 conv dim.
            swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
                apply ReLU to the Tx3x3 conv.
        """

        dim_out = int(dim_inner / 2)
        super(D_2plus1_D_bottlenck, self).__init__()
        self._eps = eps
        self._bn_mnt = bn_mmt
        self._residual = True if (dim_in == dim_out and stride != 2) else False
        self.stride = stride
        self._block_idx = block_idx
        self._swish_inner = swish_inner
        self._use_se = True if (self._block_idx + 1) % 2 else False
        self._se_ratio = se_ratio
        self._construct(dim_in, dim_inner, dim_out, stride, norm_module)

    def _construct(self, dim_in, dim_inner, dim_out, stride, norm_module):

        # point-wise 1x1x1xC conv across all channels
        self.a = nn.Sequential(
            nn.Conv3d(dim_in, dim_inner, kernel_size=[1, 1, 1], bias=False),
            norm_module(
                num_features=dim_inner, eps=self._eps, momentum=self._bn_mnt
            ),
            nn.ReLU(inplace=True),
        )

        # Depthwise (channel-separated) 3x3x3x1 conv
        # Depthwise (channel-separated) 1x3x3x1 spatial conv
        self.b1 = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=[1, 3, 3],
            stride=[1, stride, stride],
            padding=[0, 1, 1],
            groups=dim_inner,
            bias=False,
        )
        # Depthwise (channel-separated) 3x1x1x1 temporal conv
        self.b2 = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=[3, 1, 1],
            stride=[1, 1, 1],
            padding=[1, 0, 0],
            groups=dim_inner,
            bias=False,
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mnt
        )
        # Apply SE attention or not
        if self._se_ratio > 0.0 and self._use_se:
            self.se = SE(dim_inner, self._se_ratio)

        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # point-wise 1x1x1xC' conv across all channels
        self.c = nn.Sequential(
            nn.Conv3d(dim_inner, dim_out, kernel_size=1, bias=False),
            norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mnt
            ),
        )
        self.c_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        x = self.a(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b_bn(x)
        if self._use_se:
            x = self.se(x)
        x = self.b_relu(x)
        x = self.c(x)

        if self._residual:
            x = identity + x

        x = self.c_relu(x)

        return x
