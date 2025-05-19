import numpy as np
import pytest
import torch

from torch_cfd import boundaries, grids


def test_gridarray_basic_ops():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)
    c = a + b
    d = a * b
    e = a - b
    f = a / b
    assert isinstance(c, grids.GridArray)
    assert torch.allclose(c.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(d.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(e.data, torch.tensor([-3.0, -3.0, -3.0]))
    assert torch.allclose(f.data, torch.tensor([0.25, 0.4, 0.5]))


def test_gridarray_scalar_addition():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    c = a + 2.0  # __add__
    d = 2.0 + a  # __radd__
    assert torch.allclose(c.data, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(d.data, torch.tensor([3.0, 4.0, 5.0]))


def test_gridarray_scalar_multiplication():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    e = a * 2.0  # __mul__
    f = 2.0 * a  # __rmul__
    assert torch.allclose(e.data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(f.data, torch.tensor([2.0, 4.0, 6.0]))


def test_gridarray_scalar_subtraction():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([5.0, 6.0, 7.0]), offset=(0,), grid=grid)
    c = a - 2.0  # __sub__
    d = 10.0 - a  # __rsub__
    assert torch.allclose(c.data, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(d.data, torch.tensor([5.0, 4.0, 3.0]))


def test_gridarray_scalar_division():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([4.0, 8.0, 12.0]), offset=(0,), grid=grid)
    e = a / 2.0  # __truediv__
    f = 24.0 / a  # __rtruediv__
    assert torch.allclose(e.data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(f.data, torch.tensor([6.0, 3.0, 2.0]))


def test_gridarray_offset_check():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(1.0,), grid=grid)
    with pytest.raises(ValueError):
        _ = a + b


def test_gridarray_fft():
    grid = grids.Grid((4,))
    a = grids.GridArray(torch.arange(4, dtype=torch.float32), offset=(0,), grid=grid)
    fa = torch.fft.fft(a)
    assert isinstance(fa, grids.GridArray)
    assert torch.allclose(fa.data, torch.fft.fft(a.data))


def test_gridarray_to_and_clone():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = a.to(dtype=torch.float64)
    c = a.clone()
    assert b.data.dtype == torch.float64
    assert torch.allclose(c.data, a.data)


def test_gridarray_shape_dtype_properties():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    assert a.shape == (3,)
    assert a.dtype == torch.float32 or a.dtype == torch.float64


def test_gridarray_consistent_offset():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)
    assert grids.consistent_offset_arrays(a, b) == (0,)


def test_gridarray_averaged_offset():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(1.0,), grid=grid)
    avg = grids.averaged_offset_arrays(a, b)
    assert np.allclose(avg, (0.5,))


def test_gridarray_control_volume_offsets():
    data = torch.zeros((5, 5))
    grid = grids.Grid((5, 5))
    a = grids.GridArray(data, offset=(0, 0), grid=grid)
    offsets = grids.control_volume_offsets(a)
    assert offsets == ((0.5, 0), (0, 0.5))


def test_gridarray_with_batch_dim_2d():
    # Create a 2D grid with batch dimension
    grid = grids.Grid((4, 5))  # 4x5 spatial grid
    batch_size = 3

    # Create a GridArray with batch dimension [batch, x, y]
    data = torch.randn(batch_size, 4, 5)
    a = grids.GridArray(data, offset=(0, 0), grid=grid)

    assert a.shape == (3, 4, 5)
    assert a.ndim == 3
    assert a.data.shape == (3, 4, 5)

    # Test basic operations with batch dimension
    b = a + 1.0
    assert b.shape == (3, 4, 5)
    assert torch.allclose(b.data, a.data + 1.0)

    # Test multiplication with another tensor of the same shape
    c = a * a
    assert c.shape == (3, 4, 5)
    assert torch.allclose(c.data, a.data * a.data)

    # Test operations between batched arrays
    d = grids.GridArray(torch.ones_like(data), offset=(0, 0), grid=grid)
    result = a + d
    assert result.shape == (3, 4, 5)
    assert torch.allclose(result.data, a.data + 1.0)


def test_gridarray_applied():
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)
    result = grids.applied(torch.add)(a, b)
    assert isinstance(result, grids.GridArray)
    assert torch.allclose(result.data, a.data + b.data)


def test_gridarray_fft1d():
    grid = grids.Grid((8,))
    a = grids.GridArray(torch.arange(8, dtype=torch.float32), offset=(0,), grid=grid)
    fa = torch.fft.fft(a)
    assert isinstance(fa, grids.GridArray)
    assert torch.allclose(fa.data, torch.fft.fft(a.data))


def test_gridarray_ifft1d():
    grid = grids.Grid((8,))
    a = grids.GridArray(torch.arange(8, dtype=torch.float32), offset=(0,), grid=grid)
    fa = torch.fft.fft(a)
    ia = torch.fft.ifft(fa)
    assert isinstance(ia, grids.GridArray)
    # Should recover original (within numerical tolerance)
    assert torch.allclose(ia.data.real, a.data, atol=1e-6)


def test_gridarray_fft2d():
    grid = grids.Grid((4, 4))
    a = grids.GridArray(
        torch.arange(16, dtype=torch.float32).reshape(4, 4), offset=(0, 0), grid=grid
    )
    fa = torch.fft.fft2(a)
    assert isinstance(fa, grids.GridArray)
    assert torch.allclose(fa.data, torch.fft.fft2(a.data))


def test_gridarray_ifft2d():
    grid = grids.Grid((4, 4))
    a = grids.GridArray(
        torch.arange(16, dtype=torch.float32).reshape(4, 4), offset=(0, 0), grid=grid
    )
    fa = torch.fft.fft2(a)
    ia = torch.fft.ifft2(fa)
    assert isinstance(ia, grids.GridArray)
    assert torch.allclose(ia.data.real, a.data, atol=1e-6)


def test_gridarray_rfft_irfft():
    grid = grids.Grid((8,))
    a = grids.GridArray(torch.arange(8, dtype=torch.float32), offset=(0,), grid=grid)
    ra = torch.fft.rfft(a)
    assert isinstance(ra, grids.GridArray)
    ira = torch.fft.irfft(ra, n=a.data.shape[0])
    assert isinstance(ira, grids.GridArray)
    assert torch.allclose(ira.data, a.data, atol=1e-6)


def test_gridvariable_basic_ops():
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArrays
    a_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    # Create GridVariables
    a = grids.GridVariable(a_array, bc)
    b = grids.GridVariable(b_array, bc)

    # Test operations
    c = a + b
    d = a * b
    e = a - b
    f = a / b

    assert isinstance(c, grids.GridVariable)
    assert torch.allclose(c.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(d.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(e.data, torch.tensor([-3.0, -3.0, -3.0]))
    assert torch.allclose(f.data, torch.tensor([0.25, 0.4, 0.5]))

    # Verify boundary conditions are preserved
    assert c.bc == a.bc
    assert d.bc == a.bc
    assert e.bc == a.bc
    assert f.bc == a.bc


def test_gridvariable_tensor_ops():
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArray and GridVariable
    a_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a = grids.GridVariable(a_array, bc)

    # Test operations with torch.Tensor
    tensor = torch.tensor([4.0, 5.0, 6.0])

    c = a + tensor
    d = tensor + a
    e = a * tensor
    f = tensor * a
    g = a / tensor
    h = tensor / a

    assert isinstance(c, grids.GridVariable)
    assert isinstance(d, grids.GridVariable)
    assert torch.allclose(c.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(d.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(e.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(f.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(g.data, torch.tensor([0.25, 0.4, 0.5]))
    assert torch.allclose(h.data, torch.tensor([4.0, 2.5, 2.0]))

    # Verify boundary conditions are preserved
    assert c.bc == a.bc
    assert d.bc == a.bc


def test_gridvariable_gridarray_ops():
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArray and GridVariable
    a_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)
    a = grids.GridVariable(a_array, bc)

    # Test operations with GridArray
    c = a + b_array
    d = b_array + a
    e = a * b_array
    f = b_array * a
    g = a / b_array
    h = b_array / a

    assert isinstance(c, grids.GridVariable)
    assert isinstance(d, grids.GridVariable)
    assert torch.allclose(c.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(d.data, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(e.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(f.data, torch.tensor([4.0, 10.0, 18.0]))
    assert torch.allclose(g.data, torch.tensor([0.25, 0.4, 0.5]))
    assert torch.allclose(h.data, torch.tensor([4.0, 2.5, 2.0]))

    # Verify boundary conditions are preserved
    assert c.bc == a.bc
    assert d.bc == a.bc


def test_gridvariable_scalar_ops():
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridVariable
    a_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a = grids.GridVariable(a_array, bc)

    # Test operations with scalar
    c = a + 2.0
    d = 2.0 + a
    e = a * 2.0
    f = 2.0 * a
    g = a / 2.0
    h = 2.0 / a

    assert isinstance(c, grids.GridVariable)
    assert isinstance(d, grids.GridVariable)
    assert torch.allclose(c.data, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(d.data, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(e.data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(f.data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(g.data, torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(h.data, torch.tensor([2.0, 1.0, 2.0 / 3.0]))

    # Verify boundary conditions are preserved
    assert c.bc == a.bc
    assert d.bc == a.bc


def test_gridvariable_bc_check():
    grid = grids.Grid((3,))
    bc1 = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )
    bc2 = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET),)
    )

    # Create GridVariables with different boundary conditions
    a_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    a = grids.GridVariable(a_array, bc1)
    b = grids.GridVariable(b_array, bc2)

    # Test that operations between GridVariables with different BCs raise errors
    with pytest.raises(ValueError):
        _ = a + b


def test_gridarray_unary_ops():
    """Test unary operations for GridArray."""
    grid = grids.Grid((3,))
    
    # Create test GridArray
    a = grids.GridArray(torch.tensor([-1.0, 2.0, -3.0]), offset=(0,), grid=grid)
    
    # Test negation (__neg__)
    neg_a = -a
    assert isinstance(neg_a, grids.GridArray)
    assert torch.allclose(neg_a.data, torch.tensor([1.0, -2.0, 3.0]))
    assert neg_a.offset == a.offset
    assert neg_a.grid is a.grid
    
    # Test absolute value (abs)
    abs_a = abs(a)
    assert isinstance(abs_a, grids.GridArray)
    assert torch.allclose(abs_a.data, torch.tensor([1.0, 2.0, 3.0]))
    assert abs_a.offset == a.offset
    assert abs_a.grid is a.grid
    
    # Test positive (__pos__)
    pos_a = +a
    assert isinstance(pos_a, grids.GridArray)
    assert torch.allclose(pos_a.data, a.data)
    assert pos_a.offset == a.offset
    assert pos_a.grid is a.grid
    
    # Test ceil/floor
    b = grids.GridArray(torch.tensor([1.2, 2.7, 3.5]), offset=(0,), grid=grid)
    ceil_b = torch.ceil(b)
    floor_b = torch.floor(b)
    
    assert isinstance(ceil_b, grids.GridArray)
    assert isinstance(floor_b, grids.GridArray)
    assert torch.allclose(ceil_b.data, torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(floor_b.data, torch.tensor([1.0, 2.0, 3.0]))
    
    # Test round
    round_b = torch.round(b)
    assert isinstance(round_b, grids.GridArray)
    assert torch.allclose(round_b.data, torch.tensor([1.0, 3.0, 4.0]))
    
    # Test sqrt
    c = grids.GridArray(torch.tensor([4.0, 9.0, 16.0]), offset=(0,), grid=grid)
    sqrt_c = torch.sqrt(c)
    assert isinstance(sqrt_c, grids.GridArray)
    assert torch.allclose(sqrt_c.data, torch.tensor([2.0, 3.0, 4.0]))
    
    # Test exp
    exp_a = torch.exp(a)
    assert isinstance(exp_a, grids.GridArray)
    assert torch.allclose(exp_a.data, torch.exp(a.data))

def test_gridvariable_unary_ops():
    """Test unary operations for GridVariable."""
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )
    
    # Create test GridVariable
    array = grids.GridArray(torch.tensor([-1.0, 2.0, -3.0]), offset=(0,), grid=grid)
    a = grids.GridVariable(array, bc)
    
    # Test negation (__neg__)
    neg_a = -a
    assert isinstance(neg_a, grids.GridVariable)
    assert torch.allclose(neg_a.data, torch.tensor([1.0, -2.0, 3.0]))
    assert neg_a.bc is a.bc
    
    # Test absolute value (abs)
    abs_a = abs(a)
    assert isinstance(abs_a, grids.GridVariable)
    assert torch.allclose(abs_a.data, torch.tensor([1.0, 2.0, 3.0]))
    assert abs_a.bc is a.bc
    
    # Test positive (__pos__)
    pos_a = +a
    assert isinstance(pos_a, grids.GridVariable)
    assert torch.allclose(pos_a.data, a.data)
    assert pos_a.bc is a.bc


def test_gridarray_torch_functions():
    """Test various torch functions on GridArray."""
    grid = grids.Grid((3,))
    a = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    b = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)
    
    # Test sin/cos/tan
    sin_a = torch.sin(a)
    cos_a = torch.cos(a)
    tan_a = torch.tan(a)
    
    assert isinstance(sin_a, grids.GridArray)
    assert isinstance(cos_a, grids.GridArray)
    assert isinstance(tan_a, grids.GridArray)
    assert torch.allclose(sin_a.data, torch.sin(a.data))
    assert torch.allclose(cos_a.data, torch.cos(a.data))
    assert torch.allclose(tan_a.data, torch.tan(a.data))
    
    # Test max/min
    max_ab = torch.maximum(a, b)
    min_ab = torch.minimum(a, b)
    
    assert isinstance(max_ab, grids.GridArray)
    assert isinstance(min_ab, grids.GridArray)
    assert torch.allclose(max_ab.data, torch.maximum(a.data, b.data))
    assert torch.allclose(min_ab.data, torch.minimum(a.data, b.data))
    
    # Test pow
    pow_a = torch.pow(a, 2)
    assert isinstance(pow_a, grids.GridArray)
    assert torch.allclose(pow_a.data, torch.pow(a.data, 2))

def test_gridarrayvector_error_cases():
    """Test error cases for GridArrayVector operations."""
    grid = grids.Grid((3,))

    # Create GridArrays
    a1 = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2 = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    b1 = grids.GridArray(torch.tensor([2.0, 3.0, 4.0]), offset=(0,), grid=grid)

    # Create GridArrayVectors of different lengths
    avec = grids.GridArrayVector([a1, a2])
    bvec = grids.GridArrayVector([b1])

    # Test addition with vectors of different lengths
    with pytest.raises(ValueError):
        _ = avec + bvec

    # Test subtraction with vectors of different lengths
    with pytest.raises(ValueError):
        _ = avec - bvec


def test_gridarrayvector_add_operations():
    """Test addition operations (add, radd, iadd) for GridArrayVectors."""
    grid = grids.Grid((3,))

    # Create GridArrays
    a1 = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2 = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    b1 = grids.GridArray(torch.tensor([2.0, 3.0, 4.0]), offset=(0,), grid=grid)
    b2 = grids.GridArray(torch.tensor([5.0, 6.0, 7.0]), offset=(0,), grid=grid)

    # Create GridArrayVectors
    avec = grids.GridArrayVector([a1, a2])
    bvec = grids.GridArrayVector([b1, b2])

    # Test __add__
    cvec = avec + bvec
    assert isinstance(cvec, grids.GridArrayVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([9.0, 11.0, 13.0]))

    # Test __radd__ (should be the same as __add__ since it's commutative)
    cvec_r = bvec + avec
    assert isinstance(cvec_r, grids.GridArrayVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([9.0, 11.0, 13.0]))

    # Test __iadd__
    avec_copy = grids.GridArrayVector([a1.clone(), a2.clone()])
    avec_copy += bvec
    assert isinstance(avec_copy, grids.GridArrayVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([9.0, 11.0, 13.0]))


def test_gridarrayvector_sub_operations():
    """Test subtraction operations (sub, rsub, isub) for GridArrayVectors."""
    grid = grids.Grid((3,))

    # Create GridArrays
    a1 = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2 = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    b1 = grids.GridArray(torch.tensor([2.0, 3.0, 4.0]), offset=(0,), grid=grid)
    b2 = grids.GridArray(torch.tensor([5.0, 6.0, 7.0]), offset=(0,), grid=grid)

    # Create GridArrayVectors
    avec = grids.GridArrayVector([a1, a2])
    bvec = grids.GridArrayVector([b1, b2])

    # Test __sub__
    cvec = avec - bvec
    assert isinstance(cvec, grids.GridArrayVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([-1.0, -1.0, -1.0]))

    # Test __rsub__
    cvec_r = bvec - avec
    assert isinstance(cvec_r, grids.GridArrayVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([1.0, 1.0, 1.0]))

    # Test __isub__
    avec_copy = grids.GridArrayVector([a1.clone(), a2.clone()])
    avec_copy -= bvec
    assert isinstance(avec_copy, grids.GridArrayVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([-1.0, -1.0, -1.0]))


def test_gridvariablevector_add_operations():
    """Test addition operations (add, radd, iadd) for GridVariableVectors."""
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArrays
    a1_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    b1_array = grids.GridArray(torch.tensor([2.0, 3.0, 4.0]), offset=(0,), grid=grid)
    b2_array = grids.GridArray(torch.tensor([5.0, 6.0, 7.0]), offset=(0,), grid=grid)

    # Create GridVariables
    a1 = grids.GridVariable(a1_array, bc)
    a2 = grids.GridVariable(a2_array, bc)
    b1 = grids.GridVariable(b1_array, bc)
    b2 = grids.GridVariable(b2_array, bc)

    # Create GridVariableVectors
    avec = grids.GridVariableVector([a1, a2])
    bvec = grids.GridVariableVector([b1, b2])

    # Test __add__
    cvec = avec + bvec
    assert isinstance(cvec, grids.GridVariableVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([9.0, 11.0, 13.0]))

    # Test __radd__ (should be the same as __add__ since it's commutative)
    cvec_r = bvec + avec
    assert isinstance(cvec_r, grids.GridVariableVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([9.0, 11.0, 13.0]))

    # Test __iadd__ (since tuples are immutable, this creates a new object)
    avec_copy = grids.GridVariableVector(
        [
            grids.GridVariable(a1.array.clone(), a1.bc),
            grids.GridVariable(a2.array.clone(), a2.bc),
        ]
    )
    avec_copy += bvec
    assert isinstance(avec_copy, grids.GridVariableVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([3.0, 5.0, 7.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([9.0, 11.0, 13.0]))


def test_gridvariablevector_sub_operations():
    """Test subtraction operations (sub, rsub, isub) for GridVariableVectors."""
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArrays
    a1_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    b1_array = grids.GridArray(torch.tensor([2.0, 3.0, 4.0]), offset=(0,), grid=grid)
    b2_array = grids.GridArray(torch.tensor([5.0, 6.0, 7.0]), offset=(0,), grid=grid)

    # Create GridVariables
    a1 = grids.GridVariable(a1_array, bc)
    a2 = grids.GridVariable(a2_array, bc)
    b1 = grids.GridVariable(b1_array, bc)
    b2 = grids.GridVariable(b2_array, bc)

    # Create GridVariableVectors
    avec = grids.GridVariableVector([a1, a2])
    bvec = grids.GridVariableVector([b1, b2])

    # Test __sub__
    cvec = avec - bvec
    assert isinstance(cvec, grids.GridVariableVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([-1.0, -1.0, -1.0]))

    # Test __rsub__ (b - a, opposite of a - b)
    cvec_r = bvec - avec
    assert isinstance(cvec_r, grids.GridVariableVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([1.0, 1.0, 1.0]))

    # Test __isub__ (since tuples are immutable, this creates a new object)
    avec_copy = grids.GridVariableVector(
        [
            grids.GridVariable(a1.array.clone(), a1.bc),
            grids.GridVariable(a2.array.clone(), a2.bc),
        ]
    )
    avec_copy -= bvec
    assert isinstance(avec_copy, grids.GridVariableVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([-1.0, -1.0, -1.0]))

def test_gridarrayvector_mul_operations():
    """Test multiplication operations (mul, rmul, imul) for GridArrayVectors."""
    grid = grids.Grid((3,))

    # Create GridArrays
    a1 = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2 = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    # Create GridArrayVector
    avec = grids.GridArrayVector([a1, a2])
    
    # Test scalar multiplication
    # Test __mul__
    scalar = 2.0
    cvec = avec * scalar
    assert isinstance(cvec, grids.GridArrayVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([8.0, 10.0, 12.0]))

    # Test __rmul__
    cvec_r = scalar * avec
    assert isinstance(cvec_r, grids.GridArrayVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([8.0, 10.0, 12.0]))

    # Test __imul__
    avec_copy = grids.GridArrayVector([a1.clone(), a2.clone()])
    avec_copy *= scalar
    assert isinstance(avec_copy, grids.GridArrayVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([8.0, 10.0, 12.0]))
    
    # Test tensor multiplication
    tensor = torch.tensor(2.0)
    tvec = avec * tensor
    assert isinstance(tvec, grids.GridArrayVector)
    assert torch.allclose(tvec[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(tvec[1].data, torch.tensor([8.0, 10.0, 12.0]))


def test_gridarrayvector_div_operations():
    """Test division operations (truediv, rtruediv, itruediv) for GridArrayVectors."""
    grid = grids.Grid((3,))

    # Create GridArrays
    a1 = grids.GridArray(torch.tensor([2.0, 4.0, 6.0]), offset=(0,), grid=grid)
    a2 = grids.GridArray(torch.tensor([8.0, 10.0, 12.0]), offset=(0,), grid=grid)

    # Create GridArrayVector
    avec = grids.GridArrayVector([a1, a2])
    
    # Test scalar division
    # Test __truediv__
    scalar = 2.0
    cvec = avec / scalar
    assert isinstance(cvec, grids.GridArrayVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([4.0, 5.0, 6.0]))

    # Test __rtruediv__ (scalar / vector elements)
    # For rtruediv, we're testing 24.0 / vector elements
    scalar = 24.0
    cvec_r = scalar / avec
    assert isinstance(cvec_r, grids.GridArrayVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([12.0, 6.0, 4.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([3.0, 2.4, 2.0]))

    # Test __itruediv__
    avec_copy = grids.GridArrayVector([a1.clone(), a2.clone()])
    avec_copy /= 2.0
    assert isinstance(avec_copy, grids.GridArrayVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([4.0, 5.0, 6.0]))
    
    # Test tensor division
    tensor = torch.tensor(2.0)
    tvec = avec / tensor
    assert isinstance(tvec, grids.GridArrayVector)
    assert torch.allclose(tvec[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(tvec[1].data, torch.tensor([4.0, 5.0, 6.0]))


def test_gridvariablevector_mul_operations():
    """Test multiplication operations (mul, rmul, imul) for GridVariableVectors."""
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArrays
    a1_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    # Create GridVariables
    a1 = grids.GridVariable(a1_array, bc)
    a2 = grids.GridVariable(a2_array, bc)

    # Create GridVariableVector
    avec = grids.GridVariableVector([a1, a2])
    
    # Test scalar multiplication
    # Test __mul__
    scalar = 2.0
    cvec = avec * scalar
    assert isinstance(cvec, grids.GridVariableVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([8.0, 10.0, 12.0]))
    assert cvec[0].bc == a1.bc  # Verify boundary conditions are preserved
    assert cvec[1].bc == a2.bc

    # Test __rmul__
    cvec_r = scalar * avec
    assert isinstance(cvec_r, grids.GridVariableVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([8.0, 10.0, 12.0]))
    assert cvec_r[0].bc == a1.bc
    assert cvec_r[1].bc == a2.bc

    # Test __imul__ (since tuples are immutable, this creates a new object)
    avec_copy = grids.GridVariableVector([
        grids.GridVariable(a1.array.clone(), a1.bc),
        grids.GridVariable(a2.array.clone(), a2.bc),
    ])
    avec_copy *= scalar
    assert isinstance(avec_copy, grids.GridVariableVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([8.0, 10.0, 12.0]))
    assert avec_copy[0].bc == a1.bc
    assert avec_copy[1].bc == a2.bc
    
    # Test tensor multiplication
    tensor = torch.tensor(2.0)
    tvec = avec * tensor
    assert isinstance(tvec, grids.GridVariableVector)
    assert torch.allclose(tvec[0].data, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.allclose(tvec[1].data, torch.tensor([8.0, 10.0, 12.0]))


def test_gridvariablevector_div_operations():
    """Test division operations (truediv, rtruediv, itruediv) for GridVariableVectors."""
    grid = grids.Grid((3,))
    bc = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )

    # Create GridArrays
    a1_array = grids.GridArray(torch.tensor([2.0, 4.0, 6.0]), offset=(0,), grid=grid)
    a2_array = grids.GridArray(torch.tensor([8.0, 10.0, 12.0]), offset=(0,), grid=grid)

    # Create GridVariables
    a1 = grids.GridVariable(a1_array, bc)
    a2 = grids.GridVariable(a2_array, bc)

    # Create GridVariableVector
    avec = grids.GridVariableVector([a1, a2])
    
    # Test scalar division
    # Test __truediv__
    scalar = 2.0
    cvec = avec / scalar
    assert isinstance(cvec, grids.GridVariableVector)
    assert len(cvec) == 2
    assert torch.allclose(cvec[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(cvec[1].data, torch.tensor([4.0, 5.0, 6.0]))
    assert cvec[0].bc == a1.bc
    assert cvec[1].bc == a2.bc

    # Test __rtruediv__ (scalar / vector elements)
    # For rtruediv, we're testing 24.0 / vector elements
    scalar = 24.0
    cvec_r = scalar / avec
    assert isinstance(cvec_r, grids.GridVariableVector)
    assert len(cvec_r) == 2
    assert torch.allclose(cvec_r[0].data, torch.tensor([12.0, 6.0, 4.0]))
    assert torch.allclose(cvec_r[1].data, torch.tensor([3.0, 2.4, 2.0]))
    assert cvec_r[0].bc == a1.bc
    assert cvec_r[1].bc == a2.bc

    # Test __itruediv__ (since tuples are immutable, this creates a new object)
    avec_copy = grids.GridVariableVector([
        grids.GridVariable(a1.array.clone(), a1.bc),
        grids.GridVariable(a2.array.clone(), a2.bc),
    ])
    avec_copy /= 2.0
    assert isinstance(avec_copy, grids.GridVariableVector)
    assert len(avec_copy) == 2
    assert torch.allclose(avec_copy[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(avec_copy[1].data, torch.tensor([4.0, 5.0, 6.0]))
    assert avec_copy[0].bc == a1.bc
    assert avec_copy[1].bc == a2.bc
    
    # Test tensor division
    tensor = torch.tensor(2.0)
    tvec = avec / tensor
    assert isinstance(tvec, grids.GridVariableVector)
    assert torch.allclose(tvec[0].data, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(tvec[1].data, torch.tensor([4.0, 5.0, 6.0]))

def test_gridvariablevector_bc_consistency():
    """Test error cases for GridVariableVector with mismatched boundary conditions."""
    grid = grids.Grid((3,))

    bc1 = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),)
    )
    bc2 = boundaries.HomogeneousBoundaryConditions(
        ((boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET),)
    )

    # Create GridArrays
    a1_array = grids.GridArray(torch.tensor([1.0, 2.0, 3.0]), offset=(0,), grid=grid)
    a2_array = grids.GridArray(torch.tensor([4.0, 5.0, 6.0]), offset=(0,), grid=grid)

    # Create GridVariables with different boundary conditions
    a1 = grids.GridVariable(a1_array, bc1)
    a2 = grids.GridVariable(a2_array, bc1)
    b1 = grids.GridVariable(a1_array, bc2)
    b2 = grids.GridVariable(a2_array, bc2)

    # Create GridVariableVectors
    avec = grids.GridVariableVector([a1, a2])
    bvec = grids.GridVariableVector([b1, b2])

    # Test that operations between vectors with different BCs raise errors
    with pytest.raises(ValueError):
        _ = avec + bvec

    with pytest.raises(ValueError):
        _ = avec - bvec
