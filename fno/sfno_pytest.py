import pytest
import torch
import torch.fft as fft

from .sfno import (
    HelmholtzProjection,
    LiftingOperator,
    OutConv,
    SFNO,
    SpaceTimePositionalEncoding,
    SpectralConvS,
    SpectralConvT,
)
from torch_cfd.spectral import *
from contextlib import contextmanager


@contextmanager
def set_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize(
    "input_shape, modes",
    [
        ((32, 32, 5), (8, 8, 3)),
        ((64, 64, 10), (16, 16, 4)),
        ((128, 128, 20), (16, 16, 4)),
    ],
)
def test_space_time_positional_encoding_shape(input_shape, modes):
    modes_x, modes_y, modes_t = modes
    width = 16
    pe = SpaceTimePositionalEncoding(
        modes_x=modes_x,
        modes_y=modes_y,
        modes_t=modes_t,
        num_channels=width,
        input_shape=input_shape,
    )

    bsz = 2
    v = torch.randn(bsz, 1, *input_shape)
    output = pe(v)

    assert output.shape == (bsz, width, *input_shape)


@pytest.mark.parametrize("n_grid", [64, 128, 256])
def test_helmholtz_fft_mesh_2d(n_grid):
    n_grid = 64
    diam = 2 * torch.pi
    kx, ky = fft_mesh_2d(n_grid, diam)
    hzproj = HelmholtzProjection(n_grid=n_grid, diam=diam)
    assert (kx == hzproj.kx).all() and (ky == hzproj.ky).all()


@pytest.mark.parametrize("n_grid", [64, 128, 256])
def test_helmholtz_laplacian_2d(n_grid):
    diam = 2 * torch.pi
    kx, ky = fft_mesh_2d(n_grid, diam)
    lap = spectral_laplacian_2d((kx, ky))
    hzproj = HelmholtzProjection(n_grid=n_grid, diam=diam)
    assert (lap == hzproj.lap).all()


@pytest.mark.parametrize("n_grid", [64, 128, 256, 512])
def test_helmholtz_divergence_free_fp32(n_grid):
    bsz = 2
    T = 6

    hzproj = HelmholtzProjection(n_grid=n_grid)
    kx, ky = hzproj.kx, hzproj.ky
    # Create some random vector fields
    vhat = []
    for t in range(T):
        lap = hzproj.lap
        vhat_ = [
            fft.fft2(torch.randn(bsz, n_grid, n_grid)) / (5e-1 + lap) for _ in range(2)
        ]
        vhat_ = torch.stack(vhat_, dim=1)
        vhat.append(vhat_)
    vhat = torch.stack(vhat, dim=-1)

    # Apply Helmholtz projection
    w_hat = hzproj(vhat)

    # Check if result is divergence free
    div_w_hat = hzproj.div(w_hat, (kx, ky))
    div_w = fft.irfft2(div_w_hat, s=(n_grid, n_grid), dim=(1, 2)).real

    assert torch.linalg.norm(div_w) < 1e-5


@pytest.mark.parametrize("n_grid", [64, 128, 256, 512])
def test_helmholtz_divergence_free_fp64(n_grid):
    with set_default_dtype(torch.float64):
        bsz = 2
        T = 6
        diam = 2 * torch.pi
        hzproj = HelmholtzProjection(n_grid=n_grid, diam=diam, dtype=torch.float64)
        kx, ky = hzproj.kx, hzproj.ky
        lap = hzproj.lap

        # Create some random vector fields
        vhat = []
        for t in range(T):
            vhat_ = [
                fft.fft2(torch.randn(bsz, n_grid, n_grid, dtype=torch.float64))
                / (5e-1 + lap)
                for _ in range(2)
            ]
            vhat_ = torch.stack(vhat_, dim=1)
            vhat.append(vhat_)
        vhat = torch.stack(vhat, dim=-1)

        # Apply Helmholtz projection
        w_hat = hzproj(vhat)

        # Check if result is divergence free
        div_w_hat = hzproj.div(w_hat, (kx, ky))
        div_w = fft.irfft2(div_w_hat, s=(n_grid, n_grid), dim=(1, 2)).real

        assert torch.linalg.norm(div_w) < 1e-12


@pytest.mark.parametrize(
    "input_shape, modes",
    [
        ((32, 32, 5), (8, 8, 3)),
        ((64, 64, 10), (16, 16, 4)),
        ((128, 128, 20), (16, 16, 4)),
    ],
)
def test_lifting_operator_shape(input_shape, modes):
    width = 16
    modes_x, modes_y, modes_t = modes
    latent_steps = 5  # latent_steps should be <= time steps

    lifting = LiftingOperator(
        width=width,
        modes_x=modes_x,
        modes_y=modes_y,
        modes_t=modes_t,
        latent_steps=latent_steps,
    )

    bsz = 8
    nx, ny, nt = input_shape
    v = torch.randn(bsz, 1, nx, ny, nt)

    output = lifting(v)

    assert output.shape == (bsz, width, nx, ny, latent_steps)


@pytest.mark.parametrize(
    "input_shape, modes",
    [
        ((32, 32, 12), (8, 8, 3)),
        ((64, 64, 15), (16, 16, 4)),
        ((128, 128, 20), (32, 32, 10)),
    ],
)
def test_out_conv_shape(input_shape, modes):
    modes_x, modes_y, modes_t = modes
    out_dim = 1

    out_conv = OutConv(
        modes_x=modes_x,
        modes_y=modes_y,
        modes_t=modes_t,
        out_dim=out_dim,
    )

    bsz = 8
    nx, ny, nt = input_shape
    latent_steps = 10
    out_steps = 40

    v = torch.randn(bsz, out_dim, nx, ny, latent_steps)
    v_res = torch.randn(bsz, nx, ny, nt)  # Input with 5 steps

    output = out_conv(v, v_res, out_steps=out_steps)

    assert output.shape == (bsz, nx, ny, out_steps)


@pytest.mark.parametrize(
    "input_shape, modes",
    [
        ((32, 32, 12), (8, 8, 3)),
        ((64, 64, 15), (16, 16, 4)),
        ((128, 128, 20), (32, 32, 10)),
    ],
)
def test_spectral_conv_s(input_shape, modes):
    in_channels = 16
    out_channels = 16
    modes_x, modes_y, modes_t = modes

    conv = SpectralConvS(
        in_channels=in_channels,
        out_channels=out_channels,
        modes_x=modes_x,
        modes_y=modes_y,
        modes_t=modes_t,
    )

    bsz = 2
    nx, ny, nt = input_shape
    v = torch.randn(bsz, in_channels, nx, ny, nt)

    output = conv(v)

    assert output.shape == (bsz, out_channels, nx, ny, nt)


@pytest.mark.parametrize(
    "out_steps",
    [10, 20, 40],
)
def test_spectral_conv_t_with_different_out_steps(out_steps):
    in_channels = 16
    out_channels = 16
    modes_x, modes_y, modes_t = 8, 8, 4

    conv = SpectralConvT(
        in_channels=in_channels,
        out_channels=out_channels,
        modes_x=modes_x,
        modes_y=modes_y,
        modes_t=modes_t,
    )

    bsz = 2
    nx, ny, nt = 64, 64, 10

    v = torch.randn(bsz, in_channels, nx, ny, nt)

    output = conv(v, out_steps=out_steps)

    assert output.shape == (bsz, out_channels, nx, ny, out_steps)

@pytest.mark.parametrize(
    "mesh_size",
    [
        (64, 64, 10),
        (128, 128, 20),
        (256, 256, 40),
    ],
)
def test_sfno_with_different_input_sizes(mesh_size):
    modes = 8
    modes_t = 4
    width = 16
    bsz = 2
    # note: input steps >= latent steps
    model = SFNO(modes, modes, modes_t, width)

    x = torch.randn(bsz, *mesh_size)
    pred = model(x)

    # Output should match input size if out_steps is not specified
    assert pred.shape == (bsz, *mesh_size)


@pytest.mark.parametrize(
    "out_steps",
    [10, 20, 40],
)
def test_sfno_forward_with_different_output_steps(out_steps):
    modes = 8
    modes_t = 4
    width = 16

    model = SFNO(
        modes_x=modes,
        modes_y=modes,
        modes_t=modes_t,
        width=width,
        output_steps=out_steps,  # Default output steps
    )

    bsz = 2
    nx, ny, nt = 64, 64, 10
    x = torch.randn(bsz, nx, ny, nt)

    # Should use default output steps
    pred = model(x)
    assert pred.shape == (bsz, nx, ny, out_steps)
