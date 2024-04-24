# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

import math
import typing as tp
from dataclasses import dataclass, field

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    commitment_loss: tp.Optional[torch.Tensor] = None
    codebook_loss: tp.Optional[torch.Tensor] = None
    diversity_loss: tp.Optional[torch.Tensor] = None
    quantized_first: torch.Tensor = None  # the first quantized
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        ema_update (bool): Whether to use exponential moving average for updating the codebooks.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (float): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        codebook_dim: tp.Optional[int] = None,
        ema_update: bool = True,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: float = 2.0,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.log2bins = math.log2(self.bins)
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=bins,
            codebook_dim=codebook_dim,
            num_quantizers=self.n_q,
            ema_update=ema_update,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    @torch.jit.ignore
    def forward(
        self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None, n_q: int = 0
    ) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            frame_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        if not n_q:
            n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)

        quantized, codes, commitment_loss, codebook_loss, diversity_loss, quantized_first = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(
            quantized,
            codes,
            bw,
            commitment_loss=torch.mean(commitment_loss),
            codebook_loss=torch.mean(codebook_loss),
            diversity_loss=torch.mean(diversity_loss),
            quantized_first=quantized_first,
        )

    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: tp.Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth."""
        n_q = self.n_q
        if bandwidth is not None and bandwidth > 0.0:
            bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
            # bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            # bandwidth == 6.0
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, frame_rate: int):
        """Return bandwidth per quantizer for a given input frame rate.
        Each quantizer encodes a frame with lg(bins) bits.
        """
        return self.log2bins * frame_rate

    @torch.jit.export
    def encode(
        self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None, n_q: int = 0
    ) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizers to use
        and returns indices for each quantizer.
        """
        if not n_q:
            n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vq.encode(x, n_q=n_q)
        return codes

    @torch.jit.export
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized = self.vq.decode(codes)
        return quantized
