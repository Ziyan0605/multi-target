# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.autograd import Function

import model.vision.pointnet2.pytorch_utils as pt_utils


try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext  # type: ignore[attr-defined]
    _HAS_EXT = True
except ImportError:
    _ext = None  # type: ignore[assignment]
    _HAS_EXT = False

    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        # Defer to a torch-only fallback implementation instead of hard failing so
        # the repo can run without compiling the CUDA/CPP extension. This keeps the
        # behaviour consistent across machines where building the extension may not
        # be possible (e.g. headless CPU nodes).
        warnings.warn(
            "pointnet2._ext not found; falling back to slower PyTorch ops. "
            "Install the compiled extension for better performance.",
            RuntimeWarning,
        )


def _torch_furthest_point_sample(xyz, npoint):
    """Pure PyTorch furthest point sampling used as a fallback when the
    custom CUDA/CPP ops are unavailable."""

    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def _torch_gather_operation(features, idx):
    B, C, N = features.shape
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(features, 2, idx_expanded)


def _torch_grouping_operation(features, idx):
    B, C, N = features.shape
    S, K = idx.shape[1], idx.shape[2]
    features_expand = features.unsqueeze(2).expand(-1, -1, S, -1)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1, -1)
    return torch.gather(features_expand, 3, idx_expand)


def _torch_three_nn(unknown, known):
    dist = torch.cdist(unknown, known, p=2)
    dist, idx = torch.topk(dist, k=3, dim=-1, largest=False)
    return dist, idx


def _torch_three_interpolate(features, idx, weight):
    B, C, N = features.shape
    n = idx.size(1)
    features_expand = features.unsqueeze(2).expand(-1, -1, n, -1)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1, -1)
    gathered = torch.gather(features_expand, 3, idx_expand)
    weighted = gathered * weight.unsqueeze(1)
    return weighted.sum(-1)


def _torch_ball_query(radius, nsample, xyz, new_xyz):
    B, S, _ = new_xyz.shape
    N = xyz.size(1)
    device = xyz.device
    idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)

    dist = torch.cdist(new_xyz, xyz, p=2)

    for b in range(B):
        for s in range(S):
            dists = dist[b, s]
            if radius is None or math.isinf(radius):
                neighbours = torch.arange(N, device=device, dtype=torch.long)
            else:
                neighbours = torch.nonzero(dists <= radius, as_tuple=False).squeeze(-1)

            if neighbours.numel() == 0:
                neighbours = torch.topk(dists, k=1, largest=False).indices

            count = min(neighbours.numel(), nsample)
            idx[b, s, :count] = neighbours[:count]
            if count < nsample:
                idx[b, s, count:] = neighbours[count - 1]

    return idx

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


if _HAS_EXT:
    class FurthestPointSampling(Function):
        @staticmethod
        def forward(ctx, xyz, npoint):
            fps_inds = _ext.furthest_point_sampling(xyz, npoint)
            ctx.mark_non_differentiable(fps_inds)
            return fps_inds

        @staticmethod
        def backward(xyz, a=None):
            return None, None


    furthest_point_sample = FurthestPointSampling.apply

    class GatherOperation(Function):
        @staticmethod
        def forward(ctx, features, idx):
            _, C, N = features.size()
            ctx.for_backwards = (idx, C, N)
            return _ext.gather_points(features, idx)

        @staticmethod
        def backward(ctx, grad_out):
            idx, C, N = ctx.for_backwards
            grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None


    gather_operation = GatherOperation.apply

    class ThreeNN(Function):
        @staticmethod
        def forward(ctx, unknown, known):
            dist2, idx = _ext.three_nn(unknown, known)
            return torch.sqrt(dist2), idx

        @staticmethod
        def backward(ctx, a=None, b=None):
            return None, None


    three_nn = ThreeNN.apply

    class ThreeInterpolate(Function):
        @staticmethod
        def forward(ctx, features, idx, weight):
            B, c, m = features.size()
            ctx.three_interpolate_for_backward = (idx, weight, m)
            return _ext.three_interpolate(features, idx, weight)

        @staticmethod
        def backward(ctx, grad_out):
            idx, weight, m = ctx.three_interpolate_for_backward
            grad_features = _ext.three_interpolate_grad(
                grad_out.contiguous(), idx, weight, m
            )
            return grad_features, None, None


    three_interpolate = ThreeInterpolate.apply

    class GroupingOperation(Function):
        @staticmethod
        def forward(ctx, features, idx):
            B, nfeatures, nsample = idx.size()
            _, C, N = features.size()
            ctx.for_backwards = (idx, N)
            return _ext.group_points(features, idx)

        @staticmethod
        def backward(ctx, grad_out):
            idx, N = ctx.for_backwards
            grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
            return grad_features, None


    grouping_operation = GroupingOperation.apply

    class BallQuery(Function):
        @staticmethod
        def forward(ctx, radius, nsample, xyz, new_xyz):
            inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
            ctx.mark_non_differentiable(inds)
            return inds

        @staticmethod
        def backward(ctx, a=None):
            return None, None, None, None


    ball_query = BallQuery.apply
else:
    def furthest_point_sample(xyz, npoint):
        return _torch_furthest_point_sample(xyz, npoint)


    def gather_operation(features, idx):
        return _torch_gather_operation(features, idx)


    def three_nn(unknown, known):
        return _torch_three_nn(unknown, known)


    def three_interpolate(features, idx, weight):
        return _torch_three_interpolate(features, idx, weight)


    def grouping_operation(features, idx):
        return _torch_grouping_operation(features, idx)


    def ball_query(radius, nsample, xyz, new_xyz):
        return _torch_ball_query(radius, nsample, xyz, new_xyz)


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
