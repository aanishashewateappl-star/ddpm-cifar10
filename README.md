# Denoising Diffusion Probabilistic Model on CIFAR-10

A UNet-based DDPM for unconditional image generation, trained on CIFAR-10,
with visualization of the forward (noising) and reverse (denoising) processes.
The UNet architecture is adapted from the [labml.ai](https://nn.labml.ai) annotated DDPM implementation.

## Overview
- **Task:** Unconditional image generation (32x32 RGB)
- **Architecture:** UNet noise-prediction network with time embeddings and self-attention
- **Framework:** PyTorch
