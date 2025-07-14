## Diffusion Frameworks for images

A **Hydra-configured** + **PyTorch Lightning** playground for diffusion frameworks (or similar frameworks).  
Current backbones: U-Net S / M / L; planned: **Diffusion Transformers (DiT)**.

---

This repository’s layout and configuration setup are inspired by the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) by [ashleve](https://github.com/ashleve). The Lightning-Hydra-Template combines PyTorch Lightning for clean training loops with Hydra for flexible, hierarchical experiment configuration. It provides a well-organized folder structure and out-of-the-box support for rapid ML prototyping.

---

### ✨ Features
- **Python 3.10 native**
- **One-line install** – `pip install -r requirements.txt`.
- **Hydra config tree** in `configs/` (data, model, trainer, callbacks, logger, paths).
- **Lightning Modules** in `src/models/`.
- **Architectures** in `src/models/architectures/` (U-Nets now, DiT soon).
- **Two diffusion pipelines**
  - **EDM** (“Elucidating the Design Space…”)
  - **Consistency CT** (Consistency Models)
- **Batch scheduling** via `scripts/schedule.sh`.

---

### 📦 Installation

```bash
# 1️⃣ Create & activate a Python 3.10 env
conda create -n diffusion310 python=3.10
conda activate diffusion310

# 2️⃣ Clone repo and install dependencies
git clone https://github.com/your-org/diffusion_frameworks.git
cd diffusion_frameworks
pip install -r requirements.txt

# 3️⃣ Download your dataset
#    • The repo expects images of shape 3×256×256.
#    • Place them under a directory of your choice.
#      For example: ~/data/my-images

# 4️⃣ Configure environment variables
#    • Copy .env.example to .env
cp .env.example .env
#    • Open .env and set:
#      DATASET_PATH=/full/path/to/your/image/dataset
#      PROJECT_ROOT=/full/path/to/this/repo
```

### 🗂️ Directory Layout

| Path                         | Contents                                                      |
|------------------------------|---------------------------------------------------------------|
| `configs/`                   | Hydra YAMLs (data, model, trainer, callbacks, logger, paths)  |
| `src/models/`                | LightningModule implementations (EDM, Consistency CT)         |
| `src/models/architectures/`  | Backbones (U-Net S/M/L; DiT incoming)                         |
| `src/data/`                  | Lightning DataModule                                          |
| `scripts/schedule.sh`        | Simple loop for multi-config runs                             |
| `notebooks/`                 | Optional Jupyter playgrounds                                  |

### 🚀 Quick Start

#### Single training
```bash
# EDM training
python src/train.py --config-name=train_edm

# Consistency CT training
python src/train.py --config-name=train_consistencyCT
```
#### Multi training
You can train with different configs by precising them in the `scripts/schedule.sh` and then running
```bash
bash scripts/schedule.sh
```

### 🎨 Sampling
For the EDM sampling:
```python
# EDM sampling
from src.models.edm_framework import EDM
import torch

# load pretrained EDM model
model = EDM.load_from_checkpoint("path/to/last.ckpt")

# prepare inputs
latents = torch.randn(batch_size, channels, height, width)
class_labels = torch.tensor([...])  # or zeros if unconditional

# run sampling
samples = model.sampling(
    latents=latents,
    class_labels=class_labels,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float('inf'),
    S_noise=1.0,
    randn_like=torch.randn_like
)
```

For the consistency model sampling:
```python
# Consistency CT sampling
from src.models.consistencyCT import ConsistencyCT
import torch

# load pretrained ConsistencyCT model
model = ConsistencyCT.load_from_checkpoint("path/to/last.ckpt")

# prepare inputs
latents = torch.randn(batch_size, channels, height, width)
class_labels = torch.tensor([...])  # required for consistency models

# run sampling
samples = model.sampling(
    latents=latents,
    class_labels=class_labels,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    randn_like=torch.randn_like
)
```

### Don't hesitate to send me your comments/advice :)