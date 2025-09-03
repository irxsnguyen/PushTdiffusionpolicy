# Push-T Diffusion Policy

This project is a **modernized and simplified adaptation** of the [Stanford Diffusion Policy repository](https://github.com/real-stanford/diffusion-policy).  
It focuses specifically on the **state-based Push-T test environment**, refactored into a clean, standalone repo with updated dependencies and training scripts.

---

## Features
- **Environment**  
  Lightweight Push-T–style 2D block-pushing env built with `gymnasium`, `pygame`, and `pymunk`.
- **Dataset Loader**  
  - Reads demonstrations stored in Zarr format.  
  - Normalizes states and actions to `[-1, 1]`.  
  - Generates padded trajectories for configurable horizons.  
- **Diffusion Policy**  
  - Conditional 1D UNet over action sequences.  
  - Conditioning on stacked states (state-based policy).  
  - Training with DDPM + cosine noise schedule.  
  - Exponential Moving Average (EMA) weights for stable inference.  
- **Training Pipeline**  
  - PyTorch-based loop with AdamW optimizer.  
  - Cosine LR schedule with warmup.  
  - tqdm progress bars for easy monitoring.  
- **Inference**  
  - Rollout in the Push-T environment with denoised action sequences.  
  - Video logging (`vis.mp4`) and reward tracking.

---

## 🔧 Installation
```bash
git clone https://github.com/irxsnguyen/PushTdiffusionpolicy.git
cd PushTdiffusionpolicy
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
