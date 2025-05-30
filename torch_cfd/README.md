# TODO

- [x] add native PyTorch implementation for applying `torch.linalg` and `torch.fft` function directly on `GridArray` and `GridVariable` (added 0.1.).
- [x] add discrete Helmholtz decomposition (pressure projection) in both spatial and spectral domains (added 0.0.1).
- [x] adjust the functions and routines to act on `(batch, time, *spatial)` tensor, currently only `(*spatial)` is supported (added for key routines in 0.0.1).
- [x] add native FFT-based vorticity computation, instead of taking finite differences for pseudo-spectral (added in 0.0.4).
- [ ] add no-slip boundary.

# Changelog

### 0.2.0

After version `0.1.0`, I began prompt with existing codes in VSCode Copilot (using the OpenAI Enterprise API kindedly provided by UM), which arguably significantly improve the "porting->debugging->refactoring" cycle. I recorded some several good refactoring suggestions by GPT o4-mini and some by ***Claude Sonnet 3.7*** here. There were definitely over-complicated "poor" refactoring suggestions, which have been stashed after benchmarking. I found that Sonnet 3.7 is exceptionally good at providing templates for me to filling the details, when it is properly prompted with details of the functionality of current codes. Another highlight is that, based on the error or exception raised in the unittests, Sonnet 3.7 directly added configurations in `.vscode/launch.json`, saving me quite some time of copy-paste boilerplates then change by hand.

#### Major change: batch dimension for FVM
The finite volume solver now accepts the batch dimension, some key updates include
- Re-implemented flux computations of $(\boldsymbol{u}\cdot\nabla)\boldsymbol{u}$ as `nn.Module`. I originally implemented a tensor only version but did not quite work by pre-assigning `target_offsets`, which was buggy for the second component of the velocity. Sonnet 3.7 provided a very good refactoring template after being given both the original code and my implementation, after which I pretty much just fill in the blanks in [`advection.py`](./advection.py). Later I found out the bug was pretty stupid on my side, from
  ```python
  for i in range(2):
    u = v[i]
    for j in range(2):
        v[i] = flux_interpolation(u, v[j]) # offset is updated here
  ```
  to 
  ```python
  # this is gonna be buggy of course because the offset alignment will go wrong
  # the target_offsets are looped inside flux_interpolation of AdvectionVanLeer
  for offset in target_offsets:
    u = v[i]; u.offset = offset
    for j in range(2):
        v[i] = flux_interpolation(u, v[j])
        v[i].offset = offset 
  ```
  The fixed version that loops in `__call__` of [`AdvectionVanLeer` class is here](./advection.py#L451).
- Implemented a `_constant_pad_tensor` function to improve the behavior of `F.pad`, to help imposing non-homogeneous boundary conditions. It uses naturally ordered `pad` args (like Jax, unlike `F.pad`), while taking the batch dimension into consideration.
- Changed the behavior of `u.shift` taking into consideration of batch dimension. In general these methods within the `bc` class or `GridVariable` starts from the last dimension instead of the first, e.g., `for dim in range(u.grid.ndim): ...` changes to `for dim in range(-u.grid.ndim, 0): ...`.


#### Retaining only `GridVariable` class
This refactoring is suggested by ***Claude Sonnet 3.7***. In [`grids.py`](./grids.py#442), following `numpy`'s practice (see updates notes in [0.0.1](#001)) in `np.lib.mixins.NDArrayOperatorsMixin`, I originally implemented two mixin classes, [`GridArrayOperatorsMixin`](https://github.com/scaomath/torch-cfd/blob/475c7385549225570b61d8c3dcf1d415d8977f19/torch_cfd/grids.py#L304) and [`GridVariableOperatorsMixin`](https://github.com/scaomath/torch-cfd/blob/475c7385549225570b61d8c3dcf1d415d8977f19/torch_cfd/grids.py#L616) using the same boilerplate to enable operations such as `v + u` or computing the upwinding flux for two `GridArray` instances:
  ```python
  def _binary_method(name, op):
      def method(self, other):
              ...
      method.__name__ = f"__{name}__"
      return method

  class GridArrayOperatorsMixin:
  __slots__ = ()
  __lt__ = _binary_method("lt", operator.lt)
  ...

  @dataclasses.dataclass
  class GridArray(GridArrayOperatorsMixin):

  @classmethod
  def __torch_function__(self, ufunc, types, args=(), kwargs=None):
      ...
  ```
`GridVariable` is implemented largely the same recycling the codes. Note that `GridVariable` is only a container for `GridArray` that wraps boundary conditions of a field in it. Whereas`GridArray`, arguably being more vital in the whole scheme, determines an array `v`'s location by its `offset` (cell center or faces, or nodes) by Jax-CFD's original design. After a detailed prompt introducing each class's functions, after reading my workspace, **Sonnet 3.7** suggested introducing only a single `GridVariable`, while performing binary methods of two fields with the same offsets, the boundary conditions will be set to `None` if they don't share the same bc. This is already the case for some flux computations in the original `Jax-CFD` but implemented in a more hard-coded way. Now the implementation is much more concise and the boundary condition for flux computation is handled in automatically.

#### Adding a GridVectorBase class
Yet again, ***Claude Sonnet 3.7*** gave an awesome refactoring advice here. In `0.1.0`'s `grids.py`, the vector field's wrappers recycles lots of [boilerplate codes I learned from numpy back in 0.0.1](https://github.com/scaomath/torch-cfd/blob/475c7385549225570b61d8c3dcf1d415d8977f19/torch_cfd/grids.py#L801). There codes are largely the same for `GridArray` and `GridVariable` to define their behaviors when performing `__add__` and `__mul__` with a scalar, etc:
```python
class GridArrayVector(tuple):
    def __new__(cls, arrays):
        ...

    def __add__(self, other):
        ...

    __radd__ = __add__

class GridVariableVector(tuple):
    def __new__(cls, variables):
        ...

    def __add__(self, other):
        # largely the same
        ...
    __radd__ = __add__
``` 
The refactored code by Sonnet 3.7 is just amazing by cleverly exploiting the `classmethod` decorator and `super()`:
```python
from typing import TypeVar

class GridVectorBase(tuple, Generic[TypeVar("T")]):

    def __new__(cls, v: Sequence[T]):
        if not all(isinstance(x, cls._element_type()) for x in v):
            raise TypeError
        return super().__new__(cls, v)

    @classmethod
    def _element_type(cls):
        raise NotImplementedError

    def __add__(self, other):
        ...
    __radd__ = __add__


class GridVariableVector(GridVectorBase[GridVariable]):
    @classmethod
    def _element_type(cls):
        return GridVariable

```

#### Unittests
Another great feat by ***Sonnet 3.7*** is coming up with unittests using `absl.testing`'s parametrized testing. Based on [`test_grids.py`](tests/test_grids.py) I ported and tweaked by-hand example-wise, Sonnet 3.7 generated [corresponding tests using finite differences](tests/test_finite_differences.py). Even though "reasoning" regarding numerical PDE is sometimes wrong, for example, coming up with what would be shape after trimming the boundary for MAC grids variables, most are correctly formulated and helped figure out several bugs regarding the batch implementation for finite volume method.


### 0.1.0
- Implemented the FVM method on a staggered MAC grid (pressure on cell centers).
- Added native PyTorch implementation for applying `torch.linalg` and `torch.fft` functions directly on `GridArray` and `GridVariable`.
- Added native implementation of arithmetic manipulation directly on `GridVariableVector`.
- Added several helper functions `consistent_grid` to replace `consistent_grid_arrays`.
- Removed dependence of `from torch.utils._pytree import register_pytree_node`
- Minor notes:
  - Added native PyTorch dense implementation of `scipy.linalg.circulant`: for a 1d array `column`
    ```python
    # scipy version
    mat = scipy.linalg.circulant(column)

    # torch version
    idx = (n - torch.arange(n)[None].T + torch.arange(n)[None]) % n
    mat = torch.gather(column[None, ...].expand(n, -1), 1, idx)
    ```


### 0.0.8
- Starting from PyTorch 2.6.0, if data are saved using serialization (for loop with `pickle` or `dill`), then `torch.load` will raise an error, if you want to load the data, you can either add this in the imports or re-generate the data using this version.
    ```python
    torch.serialization.add_safe_globals([defaultdict])
    torch.serialization.add_safe_globals([list])
    ```

### 0.0.6
- Minor changes in function names, added `sfno` directory and moved `get_trajectory_imex` and `get_trajectory_rk4` to the data generation folder.

### 0.0.5
- added a batch dimension in solver matching. By default, the solver should work for input shapes `(batch, kx, ky)` or `(kx, ky)`. `get_trajectory()` output is either `(n_t, kx, ky)` or `(batch, n_t, kx, ky)`.


### 0.0.4
- The forcing functions are now implemented as `nn.Module` and utilize a wrapper decorator for the potential function.
- Added some common time stepping schemes, additional ones that Jax-CFD did not have includes the commonly used Crank-Nicholson IMEX.
- Combined the implementation for step size satisfying the CFL condition.


### 0.0.1
- `grids.GridArray` is implemented as a subclass of `torch.Tensor`, not the original jax implentation uses the inheritance from `np.lib.mixins.NDArrayOperatorsMixin`. `__array_ufunc__()` is replaced by `__torch_function__()`.
- The padding of `torch.nn.functional.pad()` is different from `jax.numpy.pad()`, PyTorch's pad starts from the last dimension, while Jax's pad starts from the first dimension. For example, `F.pad(x, (0, 0, 1, 0, 1, 1))` is equivalent to `jax.numpy.pad(x, ((1, 1), (1, 0), (0, 0)))` for an array of size `(*, t, h, w)`.
- A handy outer sum, which is usefully in getting the n-dimensional Laplacian in the frequency domain, is implemented as follows to replace `reduce(np.add.outer, eigenvalues)`
    ```python
    def outer_sum(x: Union[List[torch.Tensor], Tuple[torch.Tensor]]) -> torch.Tensor:
        """
        Returns the outer sum of a list of one dimensional arrays
        Example:
        x = [a, b, c]
        out = a[..., None, None] + b[..., None] + c
        """

        def _sum(a, b):
            return a[..., None] + b

        return reduce(_sum, x)
    ```