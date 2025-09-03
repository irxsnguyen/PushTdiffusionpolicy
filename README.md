# Push-T Diffusion Policy

This repository demonstrates **diffusion-based imitation learning** in the [Push-T environment](https://github.com/...), combining a physics-based simulator (PyMunk + Gymnasium) with diffusion models for long-horizon action prediction. It includes dataset preprocessing, training, evaluation, and inference pipelines.

---

## ✨ Features
- **Push-T Environment**  
  A 2D block pushing environment built with `pymunk`, `pygame`, and `gymnasium`.
- **Dataset Loader**  
  - Loads demonstrations stored in `.zarr` format.  
  - Normalizes states & actions to `[-1, 1]`.  
  - Generates padded trajectories for training.  
- **Diffusion Policy**  
  - Conditional 1D UNet predicts noise over action sequences.  
  - Training with DDPM + cosine noise schedule.  
  - EMA model weights for stable evaluation.  
- **Training Pipeline**  
  - Batched dataloaders with PyTorch.  
  - Cosine LR scheduler with warmup.  
  - Exponential Moving Average (EMA) for model stability.  
- **Inference**  
  - Rollout in the Push-T environment with diffusion denoising.  
  - Video logging & evaluation of target coverage.

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/pusht-diffusion-policy.git
cd pusht-diffusion-policy

# Python 3.8+ recommended
pip install -r requirements.txt
