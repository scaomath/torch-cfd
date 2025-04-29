# Spatiotemporal Fourier Neural Operator (SFNO)
This is a new concise implementation of the Fourier Neural Operator see [`base.py`](./base.py#L172) for a template class.

## Learning maps between Bochner spaces
SFNO now can learn a `trajectory-to-trajectory` map that inputs arbitrary-length trajectory, and outputs arbitrary-lengthed trajectory (if length is not specified, then the output length is the same with the input).

## Data generation

### FNO NSE datasdet
Generate the original FNO data where the right hand side is a fixed forcing $0.1(\sin(2\pi(x+y))+\cos(2\pi(x+y)))$.

- Training and validation data (training using first 1152 and valid using the last 128) for paper
```bash
>>> python data_gen_fno.py --num-samples 1280 --batch-size 256 --grid-size 256 --subsample 4 --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --visc 1e-3
```
  
- Test data on `256x256` grid
```bash
>>> python data_gen_fno.py --num-samples 16 --batch-size 8 --grid-size 256 --subsample 1 --double --extra-vars --time 50 --time-warmup 30 --num-steps 100 --dt 1e-3 --replicable-init --seed 42
```

### McWilliams 2d dataset

Generate the isotropic turbulence in [1] with the inverse cascade frequency signature Kolmogorov discovered.

[1]: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. Journal of Fluid Mechanics, 146, 21-43.

- Training dataset:
```bash
>>> python data_gen_McWilliams2d.py --num-samples 1152 --batch-size 128 --grid-size 256 --subsample 4 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"
```

- Testing dataset for plotting the enstrohpy spectrum in the paper
```bash
>>> python data_gen_McWilliams2d.py --num-samples 16 --grid-size 256 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"
```


## Training and evaluation scripts

### VSCode workspace for development
Please add the following setting to your VSCode workspace setting:
```json
"settings": {
		"terminal.integrated.env.osx": {"PYTHONPATH": "${workspaceFolder}"},
		"terminal.integrated.env.linux": {"PYTHONPATH": "${workspaceFolder}"},
		"jupyter.notebookFileRoot": "${workspaceFolder}",
	}
```


### Testing the arbitrary input and output discretization sizes (including time)
Run the part below `__name__ == "__main__"` in [`sfno.py`](sfno.py)
```bash
>>> python fno/sfno.py
```

### FNO NSE dataset
Train SFNO for the FNO dataset:
```bash
>>> python train.py --example "fno" --num-samples 1152 --num-val-samples 128 --epochs 10 --width 20 --modes 12 --modes-t 5 --time-steps 10 --out-time-steps 40 --beta 0.02
```

Evaluating the model only and plotting the predictions. Note for evaluation, there is no need to specify the out_steps when initializing the model. One should get around `1e-2` relative accuracy with the ground truth in 10 epochs of training, if this level is not reached, something must be wrong with the setup.
```bash
>>> python train.py --example "fno" --eval-only --epochs 10 --width 20 --modes 12 --modes-t 5  --beta 0.02 --out-time-steps 40 --demo-plots 10
```
    
### The McWilliams 2d dataset
The isotropic turbulence that has the inverse cascade of -5/3 frequency decay signature.

Train SFNO for the McWilliams2d dataset. One should get around `6e-2` relative accruacy with the ground truth after 15 epochs of training.
```bash
>>> python train.py --example "McWilliams2d" --epochs 15 --width 10 --modes 32 --modes-t 5 --beta -0.01
```

Evaluation for McWilliams2d dataset: note there will be aliasing error caused by the super-resolution when the solution is not smooth.
```bash
>>> python train.py --example "McWilliams2d" --eval-only --width 10 --modes 32 --modes-t 5 --beta -0.01 --demo-plots 10
```

## Licenses
This folder has the MIT license. Note: the license(s) in the subfolder takes precedence.

## Acknowledgments
The research of Brarda and Xi is supported by the National Science Foundation award DMS-2208412. 
The work of Li was performed under the auspices of
the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DEAC52-07NA27344 and was supported by the LLNL-LDRD program under Project No. 24ERD033. The research of Cao also is in part supported by the National Science Foundation DMS-2309778.