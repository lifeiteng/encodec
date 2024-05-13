# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings  # noqa

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils import weight_norm

from .. import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val or d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def normal_init(*shape: int):
    t = torch.empty(shape)
    nn.init.normal_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class DiversityLoss(nn.Module):
    def __init__(self, codebook_size, temperature: float = 0.9):
        super().__init__()
        self.codebook_size = codebook_size
        self.temperature = temperature

    def forward(self, confidence: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # confidence: (B * L, bins)
        point_probs = F.softmax(confidence / self.temperature, dim=1)
        class_probs = torch.mean(point_probs, dim=0)
        entropy_loss = torch.sum(class_probs * torch.log(class_probs + 1e-6))
        # consistency_loss = torch.sum(-torch.mean(point_probs * torch.log(point_probs + 1e-6), dim=1))
        return entropy_loss  # + consistency_loss


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        ema_update (bool): Whether to use exponential moving average for updating the codebooks.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (float): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        ema_update: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 2,
        affine_transform: bool = False,
    ):
        super().__init__()
        self.ema_update = ema_update
        self.decay = decay

        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("cluster_size", torch.zeros(codebook_size))

        if not kmeans_init and not ema_update:
            # make sure that the codebook_loss is used for training
            self.inited = True
            self.embed = nn.Embedding(codebook_size, dim)
            self.embed_avg = None
        else:
            self.register_buffer("inited", torch.Tensor([not kmeans_init]))
            init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
            embed = init_fn(codebook_size, dim)

            self.register_buffer("embed", embed)
            self.register_buffer("embed_avg", embed.clone())

        self.diversity_loss = DiversityLoss(codebook_size, temperature=0.9)

        self.affine_transform = affine_transform
        if affine_transform:
            self.affine_scale = nn.parameter.Parameter(torch.zeros(dim, 1))
            self.affine_bias = nn.parameter.Parameter(torch.zeros(dim, 1))

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        device = data.device
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        embed = embed.to(device)
        cluster_size = cluster_size.to(device)

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(mask[..., None], sample_vectors(samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code <= 0.0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors([self.embed])

    def preprocess(self, x):
        # x = rearrange(x, "... d -> (...) d")
        x = x.reshape([-1, x.shape[-1]])
        return x

    def quantize(self, x, topk: int = 1, indices: tp.Optional[torch.Tensor] = None):
        if self.embed_avg is None:
            embed = self.embed.weight.t()
        else:
            embed = self.embed.t()

        # affine
        if self.affine_transform:
            embed = embed * (1.0 + self.affine_scale) + self.affine_bias

        if not self.ema_update and not self.affine_transform:  # DAC
            # L2 normalize encodings and codebook (ViT-VQGAN)
            x = F.normalize(x)
            embed = F.normalize(embed, dim=0)

        if topk > 1:
            # (B*L, N)
            distance = torch.cdist(x, embed.t(), p=2)
            confidence, embed_ind = distance.topk(topk, dim=-1, largest=False)
        else:
            # (B*L, N)
            confidence = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
            embed_ind = confidence.max(dim=-1).indices

        if indices is not None:
            return confidence, indices

        return confidence, embed_ind

    def postprocess_emb(self, embed_ind: torch.Tensor, shape: tp.List[int]):
        return embed_ind.view(shape[:-1])

    def dequantize(self, embed_ind):
        if self.embed_avg is None:
            quantize = F.embedding(embed_ind, self.embed.weight)
        else:
            quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x, indices: tp.Optional[torch.Tensor] = None):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        _, embed_ind = self.quantize(x, indices=indices)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def beamsearch(self, x, num_beams: int = 2):
        distance, embed_ind = self.quantize(x, topk=num_beams)
        quantize = [self.dequantize(embed_ind[..., i]) for i in range(num_beams)]
        return distance, quantize, embed_ind

    @torch.jit.ignore
    def forward(self, x, indices: tp.Optional[torch.Tensor] = None):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        confidence, embed_ind = self.quantize(x, indices=indices)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            if self.ema_update:
                self.expire_codes_(x)
                embed_sum = x.t() @ embed_onehot
                ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
                cluster_size = (
                    laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon) * self.cluster_size.sum()
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.data.copy_(embed_normalized)

        quantize = self.dequantize(embed_ind)

        diversity_loss = self.diversity_loss(confidence)
        return quantize, embed_ind, diversity_loss


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        ema_update (bool): Whether to use exponential moving average for updating the codebooks.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (float): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        affine_transform: (bool): whether apply AffineTransform on coodbook.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        ema_update: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: float = 2.0,
        affine_transform: bool = False,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = weight_norm(nn.Linear(dim, _codebook_dim)) if requires_projection else nn.Identity()
        self.project_out = weight_norm(nn.Linear(_codebook_dim, dim)) if requires_projection else nn.Identity()
        self.epsilon = epsilon

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            ema_update=ema_update,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            affine_transform=affine_transform,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.jit.export
    def encode(self, x, indices: tp.Optional[torch.Tensor] = None):
        # x = rearrange(x, "b d n -> b n d")
        x = x.permute(0, 2, 1)
        x = self.project_in(x)
        embed_in = self._codebook.encode(x, indices=indices)
        return embed_in

    @torch.jit.export
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
        quantize = quantize.permute(0, 2, 1)
        return quantize

    def beamsearch(self, x, num_beams: int = 2):
        # x \in bxn d
        x = self.project_in(x)
        distance, quantize, embed_ind = self._codebook.beamsearch(x, num_beams=num_beams)
        quantize = torch.stack([self.project_out(q) for q in quantize], dim=-1)
        return distance, quantize, embed_ind

    @torch.jit.ignore
    def forward(self, x, indices: tp.Optional[torch.Tensor] = None):
        x = self.project_in(x)

        quantize, embed_ind, diversity_loss = self._codebook(x, indices=indices)

        # if self.training:
        #     warnings.warn('When using RVQ in training model, first check '
        #                   'https://github.com/facebookresearch/encodec/issues/25 . '
        #                   'The bug wasn\'t fixed here for reproducibility.')
        commitment_loss = F.mse_loss(quantize.detach(), x, reduction="mean")
        codebook_loss = F.mse_loss(quantize, x.detach(), reduction="mean")

        if self.training:
            quantize = x + (quantize - x).detach()

        quantize = self.project_out(quantize)
        return quantize, embed_ind, commitment_loss, codebook_loss, diversity_loss


def gather_seq(x: torch.Tensor, indices: torch.Tensor):
    bsz, seqlen, dim = x.shape
    index = torch.cumsum(torch.tensor([seqlen] * bsz).type_as(indices), dim=0).unsqueeze(dim=1)
    indices = indices + (index - seqlen)
    gathered = torch.index_select(x.contiguous().view([bsz * seqlen, dim]), dim=0, index=indices.flatten())
    return gathered.reshape([bsz, indices.shape[1], dim])


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def beamsearch(self, x: torch.Tensor, n_q: int = 2, num_beams: int = 1):
        xx = rearrange(x, "b d n -> b n d")
        D = x.shape[-1]  # D
        x = xx.reshape([-1, D])  # BxD, B=bxn
        # N = num_beams
        # BxN
        distances, quantizes, indices = self.layers[0].beamsearch(x, num_beams=num_beams)
        quantizes = quantizes.permute(0, 2, 1)
        all_indices = [indices]

        x = x.unsqueeze(1)  # BxDx1
        for q, layer in enumerate(self.layers[1:n_q]):
            residual = x - quantizes  # BxNxD
            # (BxN)xN
            distances, quantizes, _indices = layer.beamsearch(residual.reshape([-1, D]), num_beams=num_beams)
            _, mini = torch.topk(
                distances.reshape([-1, num_beams * num_beams]), num_beams, dim=1, largest=False
            )  # BxNxN -> BxN
            _indices = torch.gather(_indices.reshape([-1, num_beams * num_beams]), dim=1, index=mini)

            quantizes = quantizes.permute(0, 2, 1).reshape([-1, num_beams * num_beams, D])
            quantizes = gather_seq(quantizes, mini)

            x = residual - quantizes  # BxNxD

            # Update all_indices[-1]
            all_indices[-1] = torch.gather(all_indices[-1], dim=1, index=mini // num_beams)
            all_indices.append(_indices)

        out_indices = torch.stack(all_indices, dim=-1)[:, 0]  # (BN) Q
        return out_indices.permute(1, 0)  # [Q, BxN]

    @torch.jit.ignore
    def forward(self, x, n_q: tp.Optional[int] = None, num_beams: int = 1):
        n_q = n_q or len(self.layers)
        if num_beams > 1 and n_q > 1:
            out_indices = self.beamsearch(x, n_q, num_beams)
        else:
            out_indices = [None] * n_q

        x = rearrange(x, "b d n -> b n d")
        quantized_first, indices, commitment_loss, codebook_loss, diversity_loss = self.layers[0](
            x, indices=out_indices[0]
        )

        all_commitment_losses, all_codebook_losses, all_diversity_losses = (
            [commitment_loss],
            [codebook_loss],
            [diversity_loss],
        )
        all_indices = [indices]

        residual = x - quantized_first
        quantized_out = quantized_first

        for q, layer in enumerate(self.layers[1:n_q]):
            quantized, indices, commitment_loss, codebook_loss, diversity_loss = layer(residual, indices=out_indices[q])
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_commitment_losses.append(commitment_loss)
            all_codebook_losses.append(codebook_loss)
            all_diversity_losses.append(diversity_loss)

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x).detach()

        commitment_loss, codebook_loss, diversity_loss, out_indices = map(
            torch.stack, (all_commitment_losses, all_codebook_losses, all_diversity_losses, all_indices)
        )
        quantized_out = rearrange(quantized_out, "b n d -> b d n")
        quantized_first = rearrange(quantized_first, "b n d -> b d n")
        return quantized_out, out_indices, commitment_loss, codebook_loss, diversity_loss, quantized_first

    @torch.jit.export
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None, num_beams: int = 1) -> torch.Tensor:
        residual = x
        all_indices = []
        # n_q = n_q or len(self.layers)
        if n_q is None:
            n_q = len(self.layers)

        if num_beams > 1 and n_q > 1:
            out_indices = self.beamsearch(x, n_q, num_beams)
        else:
            out_indices = [None] * n_q

        for i in range(n_q):
            layer: VectorQuantization = self.layers[i]
            indices = layer.encode(residual, indices=out_indices[i])
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    @torch.jit.export
    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        n_q = int(q_indices.shape[0])
        for i in range(n_q):
            indices = q_indices[i]
            layer: VectorQuantization = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
