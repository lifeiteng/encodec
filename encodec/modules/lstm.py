# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

import torch
from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool = False):
        super().__init__()
        self.skip = skip
        if bidirectional:
            self.lstm = nn.LSTM(dimension, dimension // 2, num_layers, bidirectional=True)
        else:
            self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        with torch.autocast(enabled=False, device_type=x.device.type):  # LSTM doesn't support bfloat16
            dtype = x.dtype
            y, _ = self.lstm(x.to(torch.float32))
            y = y.to(dtype=dtype)

        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
