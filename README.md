# FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting

<p align="center">
<a href="https://acmmm2025.org"><img src="https://img.shields.io/badge/ACM MM-2025-FF6600.svg"></a>
<a href="https://arxiv.org/abs/2507.06671"><img src="https://img.shields.io/badge/Arxiv-2507.06671-B31B1B.svg"></a>
<a href="https://supercomputing-system-ai-lab.github.io/projects/flexgaussian/"><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
</p>

# Overview
**FlexGaussian** is a lightweight compression pipeline for **3D Gaussian Splatting (3DGS)** that:
-   Requires **no training or fine-tuning**
-   Supports **adjustable compression ratios**
-   Runs **efficiently on both desktop and mobile hardware**
    

### Key features
-   **Training-Free**: No retraining or fine-tuning required. Just plug and play.
-   **Cost-Effective**: Compresses models in under 1 minute on desktops and under 3 minutes on mobile platforms.
-   **Flexible**: With importance scores computed (a one-time cost), switching compression ratios takes 1–2 seconds.

# Installation
We tested on both desktop and mobile platforms using Ubuntu 20.04, CUDA 11.6, PyTorch 1.12.1, and Python 3.7.13.
### Install dependencies
```
# clone the repo
$ git clone git@github.com:Supercomputing-System-AI-Lab/FlexGaussian.git
$ cd FlexGaussian

# setup conda environment
$ conda env create --file environment.yml
$ conda activate FlexGaussian
```

### Setup data paths
- open `scripts/run.sh`
- specify:
	-  `source_path`: path to the dataset
	-  `model_path`: path to the pre-trained 3DGS model
	- `imp_score_path (optional)`: path to precomputed importance score in `.npz` format.

# Data preparation
### Datasets
- We have tested on Mip-NeRF360, DeepBlending, and Tanks&Temples.
- The Mip-NeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/).
- We use the Tanks&Temples and Deep Blending scenes hosted by the paper authors of 3DGS [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
- Note that while both training and test views are loaded, **importance score computation does not rely on image content**. It evaluates each 3D Gaussian's contribution by counting how many pixels it overlaps with across a view set (we use training views for simplicity, but any camera poses would work).
- The render resolution should match the training resolution.

### Pre-trained 3DGS model
- To ensure consistency and reproducibility, we use pre-trained 3DGS models released by the official 3DGS repository.
[Pre-trained Models (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
- Other models compatible with the standard 3DGS format are also supported.

# Usage
### Quick start
To run the full compression pipeline with default settings:
```
$ bash scripts/run.sh [scene_name]
```
- `scene_name` can be one of the supported scenes from Mip-NeRF360, DeepBlending, or Tanks&Temples.
- Omit `scene_name` to run compression across all 13 scenes evaluated in the paper.

### CLI usage
```
$ python compress.py -s [source_path] -m [model_path] --output_path [output_path] --eval
```
**Optional arguments**
-   `--quality_target_diff [float]`  
    Target PSNR drop. Default: `1.0` dB
    
-   `--imp_score_path [string]`  
    Path to precomputed importance score (skip score computation)
    
-   `--save_render`  
    Save rendered images from the compressed 3DGS model

# Sample output
- Below is a an example output after compressing the `bicycle` scene:
- Compressed file is compared against input 3DGS model in quality and file size.
- The pipeline time is break down to I/O, importance score calculation, and search.
```
[INFO] Running on scene: bicycle
============================================================
Best combination found: pruning=0.4, sh=0.5

[INFO] Best config:           pruning_rate = 0.4, sh_rate = 0.5
[INFO] Base PSNR:             25.1758 dB
[INFO] Best PSNR:             24.1808 dB
[INFO] PSNR drop:             0.9950 dB
[INFO] Input file size:       1450.28 MB
[INFO] Output file size:      70.97 MB
[INFO] Compression ratio:     20.43x
============================================================

============================================================
[STATS] Pipeline Time Breakdown:
  Load time:      9.01 s
  Score time:     2.97 s
  Search time:   10.31 s
  Store time:     0.24 s
  Total time:    22.52 s
============================================================
```

# Test devices
We tested on both desktop and mobile platforms using Ubuntu 20.04, CUDA 11.6, PyTorch 1.12.1, and Python 3.7.13.
**Performance may vary based on the library versions and sources of precompiled binaries.*

-   **Desktop**:
    -   GPU: Nvidia RTX 3090 (10496-cores Ampere arch)
    -   CPU: Intel Core i9-10900K (10-cores)
    -   RAM: 64 GB
-   **Mobile (Nvidia Jetson AGX Xavier)**:
    -   GPU: GV10B (512-cores Volta arch)
    -   CPU: Nvidia Carmel ARM v8 (8-cores)
    -   RAM: 16 GB Unified Memory (shared CPU/GPU)


# BibTeX
```
@inproceedings{FlexGaussian,
        title={FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting},
        author={Tian, Boyuan and Gao, Qizhe and Xianyu, Siran and Cui, Xiaotong and Zhang, Minjia},
        booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
        year={2025}
```

# Acknowledgement

-   Thanks to the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for their foundational work.
-   We also acknowledge the authors of [LightGaussian](https://lightgaussian.github.io/) for their method of calculating importance scores.
