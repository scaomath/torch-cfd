# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2024 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Tuple, Union

import torch
import torch.utils._pytree as pytree

def _normalize_axis(axis: int, ndim: int) -> int:
    """Validates and returns positive `axis` value."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"invalid axis {axis} for ndim {ndim}")
    if axis < 0:
        axis += ndim
    return axis


def slice_along_axis(
    inputs: Any, axis: int, idx: Union[slice, int], expect_same_dims: bool = True
) -> Any:
    """Returns slice of `inputs` defined by `idx` along axis `axis`.

    Args:
      inputs: tensor or a tuple of tensors to slice.
      axis: axis along which to slice the `inputs`.
      idx: index or slice along axis `axis` that is returned.
      expect_same_dims: whether all arrays should have same number of dimensions.

    Returns:
      Slice of `inputs` defined by `idx` along axis `axis`.
    """
    arrays, tree_def = pytree.tree_flatten(inputs)
    ndims = set(a.ndim for a in arrays)
    if expect_same_dims and len(ndims) != 1:
        raise ValueError(
            "arrays in `inputs` expected to have same ndims, but have "
            f"{ndims}. To allow this, pass expect_same_dims=False"
        )
    sliced = []
    for array in arrays:
        ndim = array.ndim
        slc = tuple(
            idx if j == _normalize_axis(axis, ndim) else slice(None)
            for j in range(ndim)
        )
        sliced.append(array[slc])
    return pytree.tree_unflatten(tree_def, sliced)


def split_along_axis(
    inputs: Any, split_idx: int, axis: int, expect_same_dims: bool = True
) -> Tuple[Any, Any]:
    """Returns a tuple of slices of `inputs` split along `axis` at `split_idx`.

    Args:
      inputs: pytree of arrays to split.
      split_idx: index along `axis` where the second split starts.
      axis: axis along which to split the `inputs`.
      expect_same_dims: whether all arrays should have same number of dimensions.

    Returns:
      Tuple of slices of `inputs` split along `axis` at `split_idx`.
    """

    first_slice = slice_along_axis(inputs, axis, slice(0, split_idx), expect_same_dims)
    second_slice = slice_along_axis(
        inputs, axis, slice(split_idx, None), expect_same_dims
    )
    return first_slice, second_slice


def split_axis(inputs: Any, dim: int, keep_dims: bool = False) -> Tuple[Any, ...]:
    """Splits the arrays in `inputs` along `axis`.

    Args:
      inputs: pytree to be split.
      axis: axis along which to split the `inputs`.
      keep_dims: whether to keep `axis` dimension.

    Returns:
      Tuple of pytrees that correspond to slices of `inputs` along `axis`. The
      `axis` dimension is removed if `squeeze is set to True.

    Raises:
      ValueError: if arrays in `inputs` don't have unique size along `axis`.
    """
    arrays, tree_def = pytree.tree_flatten(inputs)
    axis_shapes = set(a.shape[dim] for a in arrays)
    if len(axis_shapes) != 1:
        raise ValueError(f"Arrays must have equal sized axis but got {axis_shapes}")
    (axis_shape,) = axis_shapes
    splits = [torch.split(a, axis_shape, dim=dim) for a in arrays]
    if not keep_dims:
        splits = pytree.tree_map(lambda a: torch.squeeze(a, dim), splits)
    splits = zip(*splits)
    return tuple(pytree.tree_unflatten(tree_def, leaves) for leaves in splits)


