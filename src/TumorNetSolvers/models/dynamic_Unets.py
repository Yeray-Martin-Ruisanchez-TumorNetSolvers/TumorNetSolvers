# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Copyright 2022 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Modified by Zeineb Haouari on December 5, 2024
# This file has been modified from its original version. Code adapted from:
# - nnUnet (https://github.com/MIC-DKFZ/nnUNet)
# - Dynamic Network Architectures (https://github.com/MIC-DKFZ/dynamic-network-architectures)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Contains definitions of extended dynamic Unets to account for the integration of biophysical 
param vector at the bottleneck
"""
import pydoc
import warnings
from typing import Union, List, Tuple, Type
import numpy as np
import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from batchgenerators.utilities.file_and_folder_operations import join

class UNetDecoder(nn.Module):
    """
    Implements a U-Net decoder with support for parameter integration and deep supervision.

    Args:
        encoder: Encoder model providing skip connections.
        num_classes: Number of output classes.
        n_conv_per_stage: Number of convolutions per stage.
        deep_supervision: Enables deep supervision.
        nonlin_first: Non-linearity is applied before normalization.
        norm_op, norm_op_kwargs: Normalization operation and its parameters.
        dropout_op, dropout_op_kwargs: Dropout operation and its parameters.
        nonlin, nonlin_kwargs: Non-linearity and its parameters.
        conv_bias: Bias in convolution layers.
        param_dim: Dimension of parameter vector for bottleneck integration.
    """
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None, 
                 param_dim: int =5
                 ):
        super().__init__()
        self.param_dim=param_dim
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # Initialize lists to store layers
        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            # Adjust the number of input features for the first stage
            if s == 1:
                input_features_below = encoder.output_channels[-s] + param_dim
            else:
                input_features_below = encoder.output_channels[-s]
            
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            
            # Define the transpose convolution layer
            transpconv_layer = transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            )
            transpconvs.append(transpconv_layer)
            
            # Define the stacked convolution blocks
            conv_blocks = StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            )
            stages.append(conv_blocks)
            
            # Define the segmentation layer
            seg_layer = encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            seg_layers.append(seg_layer)

        # Convert lists to ModuleList and move to GPU if necessary
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output




class PlainConvUNetNew(nn.Module):
    """
    Implements a U-Net architecture with parameter integration at the bottleneck.

    Args:
        input_channels: Number of input channels.
        n_stages: Number of stages in the encoder.
        features_per_stage: Features at each stage.
        conv_op, kernel_sizes, strides: Convolution operation and parameters.
        n_conv_per_stage: Convolutions per encoder stage.
        num_classes: Number of output classes.
        n_conv_per_stage_decoder: Convolutions per decoder stage.
        conv_bias, norm_op, norm_op_kwargs: Convolution and normalization parameters.
        dropout_op, dropout_op_kwargs: Dropout operation and parameters.
        nonlin, nonlin_kwargs: Non-linearity and its parameters.
        deep_supervision: Enables deep supervision.
        nonlin_first: Non-linearity is applied before normalization.
        param_dim: Dimension of parameter vector.
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Dropout]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 param_dim: int =5,
                 inputs_shape: torch.Size = None
                 ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)
        self.param_dim=param_dim
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        
        latent_spatial_size = list(inputs_shape[-3:])  # e.g., [64, 64, 64]
        for stride in self.encoder.strides:
            latent_spatial_size = [i // s for i, s in zip(latent_spatial_size, stride)]
        self.latent_space_sz = latent_spatial_size[0]

        self.param_fc = nn.Linear(
            in_features=param_dim,
            out_features=self.latent_space_sz ** 3 * param_dim
        )

        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first, param_dim=self.param_dim)

    def integrateParams(self, param, latent_space_sz, skips, batch_size):
        param = param.to(skips[-1].device)  # Ensure param is on the same device as the skips
        self.param_fc = self.param_fc.to(skips[-1].device)
        p = self.param_fc(param).view(batch_size, param.size(1), latent_space_sz, latent_space_sz, latent_space_sz)
        # Concatenate along channel dimension
        z_cat = torch.cat((skips[-1], p), dim=1)
        skips[-1] = z_cat

    def forward(self, x, param):
        batch_size = x.size(0)
        skips = self.encoder(x)
        latent_space_sz = skips[-1].shape[-1]
        self.integrateParams(param, latent_space_sz, skips, batch_size)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)



def get_network_from_plans_new(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels, inputs_shape,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    architecture_kwargs = dict(**arch_kwargs)
    architecture_classes = {
    'PlainConvUnetNew': PlainConvUNetNew
    }
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = architecture_classes[arch_class_name]
    # sometimes things move around, this makes it so that we can at least recover some of that

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision
    
    

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        inputs_shape=inputs_shape,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

