import torch
from fno.sfno import SFNO


if __name__ == "__main__":
    """
    testing the arbitrary sizes inference for both
    spatial and temporal dimensions of SFNO
    """
    modes = 8
    modes_t = 2
    width = 10
    bsz = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sizes = [(n, n, n_t) for (n, n_t) in zip([64, 128, 256], [5, 10, 20])]
    model = SFNO(modes, modes, modes_t, width, 
                 latent_steps=3).to(device)
    x = torch.randn(bsz, *sizes[0]).to(device)
    _ = model(x)

    try:
        from torchinfo import summary

        """
        torchinfo has not resolve the complex number problem
        """
        summary(model, input_size=(bsz, *sizes[-1]))
    except:
        raise ImportError(
            "torchinfo is not installed, please install it to get the model summary"
        )
    del model

    print("\n" * 3)
    for k, size in enumerate(sizes):
        torch.cuda.empty_cache()
        model = SFNO(modes, modes, modes_t, width, latent_steps=3).to(device)
        model.add_latent_hook("activations")
        x = torch.randn(bsz, *size).to(device)
        pred = model(x)
        print(f"\n\ninput shape:  {list(x.size())}")
        print(f"output shape: {list(pred.size())}")
        for k, v in model.latent_tensors.items():
            print(k, list(v.shape))
        del model

    print("\n")
    # test evaluation speed
    from time import time

    torch.cuda.empty_cache()
    model = SFNO(modes, modes, modes_t, width, latent_steps=3).to(device)
    model.eval()
    x = torch.randn(bsz, *sizes[1]).to(device)
    start_time = time()
    for _ in range(100):
        pred = model(x)
    end_time = time()
    print(f"Average eval for time: {(end_time - start_time) / 100:.6f} seconds")
    del model