import torch
import torch.nn as nn
from torch.nn import functional as F

from model_layers import (
    ConvBlock,
    ResidualBlock,
    ConvolutionalPolicyHead,
    ConvolutionalValueOrMovesLeftHead,
)
import torch
from torch import nn
from collections import OrderedDict
from typing import Optional, NamedTuple
from math import prod, sqrt


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor

class EasyLeela(nn.Module):
    def __init__(
        self,
        num_filters,
        num_residual_blocks,
        se_ratio,
        policy_loss_weight,
        value_loss_weight,
        learning_rate
    ):
        super().__init__()
        self.input_block = ConvBlock(
            input_channels=19, filter_size=3, output_channels=num_filters
        )
        residual_blocks = OrderedDict(
            [
                (f"residual_block_{i}", ResidualBlock(num_filters, se_ratio))
                for i in range(num_residual_blocks)
            ]
        )
        self.residual_blocks = nn.Sequential(residual_blocks)
        self.policy_head = ConvolutionalPolicyHead(num_filters=num_filters)
        # The value head has 3 dimensions for estimating the likelihood of win/draw/loss (WDL)
        self.value_head = ConvolutionalValueOrMovesLeftHead(
            input_dim=num_filters,
            output_dim=3,
            num_filters=32,
            hidden_dim=128,
            relu=False, #TODO: check that the likelihood is all positive 
        )
  
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.learning_rate = learning_rate

    def forward(self, input_planes: torch.Tensor) -> ModelOutput:
        flow = input_planes.reshape(-1, 19, 8, 8)
        flow = self.input_block(flow)
        flow = self.residual_blocks(flow)
        policy_out = self.policy_head(flow)
        # expect it to be in [-1,1]
        policy_out = torch.tanh(policy_out)

        value_out = self.value_head(flow)
        #expect it to be in [0,1]
        value_out = F.softmax(value_out, dim=1)
        return ModelOutput(policy_out, value_out)
