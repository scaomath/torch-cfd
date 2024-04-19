## TODO

- [ ] add native PyTorch implementation for applying `torch.linalg` and `torch.fft` function directly on `GridArray`.
- [ ] add discrete Helmholtz decomposition in both spatial and spectral domains.
- [ ] adjust the function to act on `(batch, time, *spatial)` tensor, currently only `(*spatial)` is supported.

## Changelog
- `grids.GridArray` is implemented as a subclass of `torch.Tensor`, not the original jax implentation uses the inheritance from `np.lib.mixins.NDArrayOperatorsMixin`. `__array_ufunc__()` is replaced by `__torch_function__()`.
- The padding of `torch.nn.functional.pad()` is different from `jax.numpy.pad()`, PyTorch's pad starts from the last dimension, while Jax's pad starts from the first dimension. For example, `F.pad(x, (0, 0, 1, 0, 1, 1))` is equivalent to `jax.numpy.pad(x, ((1, 1), (1, 0), (0, 0)))` for an array of size `(*, t, h, w)`.