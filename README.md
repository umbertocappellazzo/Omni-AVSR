# 🎯 Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2024.12345-b31b1b.svg)](https://arxiv.org/abs/your-paper-id)
[![Website](https://img.shields.io/badge/🌐-Website-blue.svg)](https://umbertocappellazzo.github.io/Omni-AVSR/)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=umbertocappellazzo.Omni-AVSR)](https://github.com/umbertocappellazzo/Omni-AVSR)
[![GitHub Stars](https://img.shields.io/github/stars/umbertocappellazzo/Omni-AVSR?style=social)](https://github.com/umbertocappellazzo/Omni-AVSR/stargazers)

**[Umberto Cappellazzo¹](#) · [Xubo Liu²](#) · [Pingchuan Ma¹](#) · [Stavros Petridis¹](#) · [Maja Pantic¹](#)**

¹Imperial College London ²University of Surrey

### 📄 [`Paper`](https://arxiv.org/abs/your-paper-id) | 🌐 [`Project Page`](https://github.com/umbertocappellazzo/Omni-AVSR) | 💻 [`Code`](https://github.com/umbertocappellazzo/Omni-AVSR) | 🔖 [`BibTeX`](#-citation)

</div>

---

## 📢 News

- **[2024-11]** 🚀 Code and models released!
- **[2024-10]** 📝 Paper submitted to arXiv.

---

## 🌟 Highlights

<div align="center">
  <img src="assets/results.png" alt="Teaser" width="800"/>
  <p><i>Figure 1: Overview of our method and main results.</i></p>
</div>

✨ **Key Contributions:**
- We present **Omni-AVSR**, the first audio-visual LLM that supports ASR, VSR, and AVSR jointly while enabling elastic inference under a single set of weights.
- Omni-AVSR hinges upon an optimized *matryoshka*-based framework to support efficient multi-granularity training.
- To adapt the backbone LLM to all tasks in a parameter-efficient manner, Omni-AVSR uses three ad-hoc *LoRA*-based methods.
- Omni-AVSR achieves SoTA results on LRS2 and LRS3 benchmarks, whilst substantially reducing training and deployment costs.

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Method Overview](#-method-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Pretrained Models](#-pretrained-models)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 📝 Abstract

> **Omni-AVSR** is a unified multimodal large language model designed to perform auditory (ASR), visual (VSR), and audio-visual (AVSR) speech recognition within a single framework. It integrates **efficient Matryoshka-based multi-granularity training**, enabling flexible control of token compression rates for elastic inference across modalities. To adapt the LLM efficiently, Omni-AVSR introduces **Omni-LoRA**, offering shared, task-specific, and hybrid fine-tuning strategies. Experiments on LRS2 and LRS3 show that Omni-AVSR matches or surpasses state-of-the-art results while drastically reducing training and deployment costs.

---

## 🔬 Method Overview

### Architecture

<div align="center">
  <img src="assets/pipeline.png" alt="Architecture" width="800"/>
  <p><i>Figure 2: Overall architecture of our proposed Omni-AVSR method.</i></p>
</div>

Our method consists of three main components:

1. **Component A**: Description of the first component and its role
2. **Component B**: Description of the second component and its role
3. **Component C**: Description of the third component and its role

### Key Features

- 🎨 **Feature 1**: Detailed explanation
- ⚡ **Feature 2**: Detailed explanation
- 🔧 **Feature 3**: Detailed explanation

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Create a conda environment
conda create -n yourproject python=3.8
conda activate yourproject

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with:

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
tqdm>=4.62.0
tensorboard>=2.8.0
pyyaml>=6.0
matplotlib>=3.5.0
```

---

## 🚀 Quick Start

### Demo

Run a quick demo with our pretrained model:

```bash
# Download pretrained model
bash scripts/download_models.sh

# Run inference on sample data
python demo.py --input samples/input.jpg --output results/output.jpg --model checkpoints/model_best.pth
```

### Expected Output

```
Loading model from checkpoints/model_best.pth
Processing input.jpg...
✓ Result saved to results/output.jpg
Inference time: 0.05s
```

---

## 📊 Dataset Preparation

### Supported Datasets

We support the following datasets:

- **Dataset 1**: [Download Link](https://dataset1-url.com)
- **Dataset 2**: [Download Link](https://dataset2-url.com)
- **Dataset 3**: [Download Link](https://dataset3-url.com)

### Data Structure

Organize your data as follows:

```
data/
├── dataset1/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── dataset2/
    └── ...
```

### Preprocessing

```bash
# Preprocess the dataset
python scripts/preprocess.py --dataset dataset1 --data_root data/dataset1
```

---

## 🎓 Training

### Train from Scratch

```bash
# Single GPU training
python train.py --config configs/default.yaml --gpu 0

# Multi-GPU training (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --config configs/default.yaml \
    --distributed
```

### Configuration

Edit `configs/default.yaml` to customize training parameters:

```yaml
# Model
model:
  type: YourModel
  backbone: resnet50
  num_classes: 10

# Training
train:
  batch_size: 32
  epochs: 100
  lr: 0.001
  optimizer: Adam
  scheduler: CosineAnnealingLR

# Data
data:
  dataset: dataset1
  train_split: train
  val_split: val
  num_workers: 4
```

### Resume Training

```bash
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

---

## 📈 Evaluation

### Evaluate on Test Set

```bash
# Evaluate with pretrained model
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/model_best.pth \
    --split test
```

### Metrics

Our evaluation includes:

- **Metric 1**: Description
- **Metric 2**: Description  
- **Metric 3**: Description

---

## 🏆 Results

### Main Results

Performance on [Dataset Name] test set:

| Method | Metric 1 | Metric 2 | Metric 3 | Params (M) |
|--------|----------|----------|----------|------------|
| Baseline A | 75.2 | 0.85 | 2.1s | 25.5 |
| Baseline B | 78.5 | 0.88 | 1.8s | 32.1 |
| **Ours** | **85.3** | **0.92** | **1.5s** | **28.3** |

### Qualitative Results

<div align="center">
  <img src="assets/results_comparison.png" alt="Results" width="800"/>
  <p><i>Figure 3: Qualitative comparison with baseline methods.</i></p>
</div>

### Ablation Studies

| Component | Metric 1 | Metric 2 |
|-----------|----------|----------|
| Baseline | 75.2 | 0.85 |
| + Component A | 80.1 | 0.89 |
| + Component B | 83.5 | 0.91 |
| + Component C (Full) | **85.3** | **0.92** |

---

## 🎁 Pretrained Models

Download our pretrained models:

| Model | Dataset | Performance | Download |
|-------|---------|-------------|----------|
| Model-Base | Dataset1 | 85.3% | [Link](https://drive.google.com/your-link) |
| Model-Large | Dataset1 | 87.1% | [Link](https://drive.google.com/your-link) |
| Model-Base | Dataset2 | 82.5% | [Link](https://drive.google.com/your-link) |

Place downloaded models in `checkpoints/` directory.

---

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{yourname2024yourtitle,
  title={Your Paper Title Here},
  author={Your Name and Co-Author Name},
  journal={arXiv preprint arXiv:2024.12345},
  year={2024}
}
```

---

## 🙏 Acknowledgements

- Code relies on [auto-avsr](https://github.com/mpc001/auto_avsr), [avhubert](https://github.com/facebookresearch/av_hubert), and [Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR) repositories
- Built with [PyTorch](https://pytorch.org/)

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://github.com/umbertocappellazzo/Omni-AVSR) and our [preprint]()

---

<div align="center">
