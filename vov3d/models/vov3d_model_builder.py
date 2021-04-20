#!/usr/bin/env python3
# Copyright Youngwan Lee (ETRI). All Rights Reserved.

"""Video models."""

import math
import torch.nn as nn

import vov3d.utils.weight_init_helper as init_helper
from vov3d.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY


# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "vov3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ]
}

_POOL1 = {
    "vov3d": [[1, 1, 1]],
}


# Number of blocks for different stages given the model depth.
# 31 - medium | 70 - large model, respectively.
_VOV3D_STAGE_DEPTH = {31: (1, 2, 3, 2), 70: (1, 2, 5, 3)}


@MODEL_REGISTRY.register()
class VoV3D(nn.Module):
    """
    VOV3D3D model builder. It builds a VOV3D.

    Youngwan Lee.
    "Diverse Temporal Aggregation and Depthwise Spatiotemporal Factorization
    for Efficient Video Classification"
    https://arxiv.org/abs/2012.00317
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VoV3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway VoV3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        num_blocks = _VOV3D_STAGE_DEPTH[cfg.VOV3D.DEPTH]
        num_groups = cfg.VOV3D.NUM_CONV
        inner_ch = cfg.VOV3D.INNER_CH
        dim_stem = int(inner_ch / 2)
        inner_multiplier = cfg.VOV3D.INNER_MULTIPLIER
        out_multiplier = cfg.VOV3D.OUT_MULTIPLIER

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_stem],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        dim_in = dim_stem
        for stage in range(4):
            prefix = "s{}".format(stage + 2)
            dim_out = int(inner_ch * out_multiplier[stage])
            dim_inner = int(inner_ch * inner_multiplier[stage])

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=cfg.VOV3D.SPATIAL_STRIDES[0],
                num_blocks=[num_blocks[stage]],
                num_groups=[num_groups],
                num_block_temp_kernel=[num_blocks[stage]],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.VOV3D.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                inplace_relu=cfg.RESNET.INPLACE_RELU,
                dilation=cfg.VOV3D.TEMPORAL_DILATIONS[0],
                norm_module=self.norm_module,
            )

            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[inner_ch * out_multiplier[-1]],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=inner_ch * out_multiplier[-1],
                dim_inner=inner_ch * inner_multiplier[-1],
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x
