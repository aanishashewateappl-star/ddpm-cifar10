# Denoising Diffusion Probabilistic Model on CIFAR-10

A UNet-based DDPM for unconditional image generation, trained on CIFAR-10,
with visualization of the forward (noising) and reverse (denoising) processes.
The UNet architecture is adapted from the [labml.ai](https://nn.labml.ai) annotated DDPM implementation.

## Overview
- **Task:** Unconditional image generation (32x32 RGB)
- **Architecture:** UNet noise-prediction network with time embeddings and self-attention
- **Framework:** PyTorch
- **Evaluation:** FID (pytorch-fid)

## Approach
1. **Forward process:** progressively adds Gaussian noise to training images; x_t can be sampled in closed form from x_0 using the cumulative product of alphas
2. **Reverse process:** a UNet is trained to predict the noise added at a random timestep (MSE loss)
3. **Sampling:** start from pure Gaussian noise and iteratively denoise for T steps
4. **Training details:**
   - Timesteps: T = 1000, linear beta schedule from 1e-4 to 0.04
   - UNet: base channels 32, channel multipliers (1, 2, 2, 4), 2 blocks per resolution, attention at the two lowest resolutions
   - Optimizer: Adam, lr 7e-5, batch size 128, 30 epochs
   - Data: 50,000 CIFAR-10 training images scaled to [-1, 1]
   - Mixed precision (fp16) training on a Colab T4 GPU

## Tech Stack
PyTorch · Hugging Face `datasets` · pytorch-fid · Google Colab (T4 GPU)
