
# MambaFuse: Linear-Time State Space Framework for Cross-Modality IR-VIS Image Fusion

[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-LLVIP-blue.svg)](#dataset-preparation)
[![Parameters](https://img.shields.io/badge/Parameters-0.08_MB-success.svg)](#1-hardware-efficiency-swap-constraints)

This repository contains the official PyTorch implementation for the term project: **MambaFuse: A Linear-Time State Space Framework for Cross-Modality Infrared-Visible Image Fusion on the LLVIP Dataset**.

## 📌 Project Overview
Infrared-visible image fusion (IVIF) is essential for robust Unmanned Aerial Vehicle (UAV) perception. However, airborne platforms operate under strict Size, Weight, and Power (SWaP) constraints. Traditional Convolutional Neural Networks (CNNs) lack global context, while Vision Transformers (ViTs) suffer from prohibitive $O(N^2)$ quadratic computational complexity. 

**MambaFuse** addresses these limitations by leveraging a full-precision State Space Model (SSM) sequence block. It dynamically aligns and fuses cross-modal features in linear time ($O(N)$) without relying on heavy attention matrices. To ensure robust generalization and prevent data leakage, the model is evaluated using a strict train/test split on the extensive **LLVIP dataset** (>15,000 aligned image pairs).

### Key Features
* **Ultra-Lightweight Architecture:** Requires less than 100 Kilobytes of memory footprint and utilizes roughly 70% fewer FLOPs than standard fusion baselines, making it optimally suited for edge AI accelerators (e.g., NVIDIA Jetson Nano).
* **Linear-Time Sequence Fusion:** Replaces quadratic self-attention with a dynamic Mamba (SSM) selective scan to globally fuse thermal and visible tokens in the latent space.
* **Strict Train/Test Validation:** The codebase explicitly isolates training data (12,025 pairs) from testing data (3,463 unseen pairs) to ensure legitimate, leak-free evaluation.

---

## 🚀 Getting Started

### Prerequisites
The code is designed to run in standard local Python environments or Google Colab (A100/L4 recommended for full-scale training).
* Python 3.8+
* PyTorch 2.0+
* OpenCV (`cv2`)
* NumPy, Matplotlib, scikit-image
* `thop` (for Hardware FLOPs profiling)

Install the required packages using:
```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-image thop

```

### Dataset Preparation

This project utilizes the [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP).
Structure your downloaded dataset directory strictly with train/test splits as follows:

```text
dataset/
├── visible/
│   ├── train/    # .jpg visible training images
│   └── test/     # .jpg visible testing images
└── infrared/
    ├── train/    # .jpg infrared training images
    └── test/     # .jpg infrared testing images

```

*Note for Colab Users:* To bypass Google Drive network bottlenecks during training, it is highly recommended to zip the dataset, copy it to Colab's local `/content/` storage, and unzip it locally before running the training loop.

---

## 💻 Usage

To run the full pipeline (including architecture setup, hardware profiling, full-scale training, and test set evaluation), execute the main script:

```bash
python train_and_eval.py

```

### Pipeline Breakdown

1. **`RealLLVIPDataset`**: Custom PyTorch DataLoader handling the explicit Train/Test modality splits and tensor transformations.
2. **`SimplifiedSSMLayer`**: A stable, pure-PyTorch implementation of the linear differential state-space evolution equations.
3. **`MambaFusionBlock`**: Flattens, sums, and performs the selective sequence scan over the multimodal tokens.
4. **Hardware Profiler**: Automatically calculates and outputs Parameters (MB) and FLOPs using `thop` for MambaFuse and baseline models.
5. **Metrics Evaluator**: Calculates Entropy (EN), Spatial Frequency (SF), Standard Deviation (SD), and Structural Similarity (SSIM) on the unseen test split.

---

## 📊 Experimental Results

### 1. Hardware Efficiency (SWaP Constraints)

MambaFuse was explicitly engineered for UAV deployment. Profiling via `thop` confirms it requires significantly fewer FLOPs than established spatial fusion networks like DenseFuse, actively preventing battery drain and thermal throttling on edge devices.

| Architecture | Model Size (MB) | Computation (GFLOPs) | Suitability for UAV Edge Deployment |
| --- | --- | --- | --- |
| Standard ViT (Reference) | ~86.0000 | ~15.5000 | Prohibitive (High Latency/Power) |
| DenseFuse (Established) | 0.0096 | 1.0853 | Moderate |
| CNN-Concat Baseline | 0.2450 | 0.8140 | Moderate |
| **MambaFuse (Proposed)** | **0.0868** | **0.3057** | **Optimal (Ultra-Lightweight)** |

### 2. Full-Scale Fusion Quality Metrics

The framework was evaluated against a lightweight CNN spatial concatenation baseline. Both models were trained for 20 epochs across the full 12,025-pair LLVIP training split and evaluated on 3,463 completely unseen test pairs.

| Model Configuration | Entropy (EN) ↑ | Spatial Freq. (SF) ↑ | Std. Dev (SD) ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- |
| CNN-Concat Baseline | 6.4262 | 4.9263 | 23.3846 | 0.4381 |
| **MambaFuse (Ours)** | **6.4612** | **5.1076** | **24.0127** | **0.4387** |

> **Analysis:** While localized CNN baselines can superficially match sequence models on small data subsets due to rapid texture memorization, the full-scale deployment demonstrates the true capability of the State Space Model. Given the full 12,025 training pairs, the global sequence modeling of MambaFuse mathematically converges, outperforming the baseline across all structural and variance-based metrics (EN, SF, SD, and SSIM) while operating at a fraction of the computational footprint.

### 3. Qualitative Visual Output

*(Refer to `publication_figure.png` in the repository)*
Visual analysis on the test set confirms MambaFuse selectively scans the latent space to retain details from both modalities seamlessly. It successfully injects the prominent thermal footprint of active targets (e.g., pedestrians) into the structural visible background without introducing the severe halo artifacts or edge degradation often caused by simple spatial pooling.

---

## 📝 Term Project Context

This repository was created as the final deliverable for the course term project. The methodology evolved from initial binarized straight-through estimator approximations into a reliable, full-precision State Space structural setup optimized specifically for edge-deployable SWaP constraints and validated at full dataset scale.

**Acknowledgments:**

* The base mathematical formulation of SSMs is inspired by *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao, 2023).
* The dataset utilized is *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision* (Jia et al., 2021).

```

```
