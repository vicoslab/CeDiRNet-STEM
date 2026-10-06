# CeDiRNet-STEM: Center Direction Network for Nanoparticles detection in STEM images

## Installation

Dependency:
 * Python >= 3.8
 * PyTorch >= 1.9
 * [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) and [timm](https://pypi.org/project/timm/)
 * opencv-python
 * numpy, scipy, scikit_image, scikit_learn

See `requirements.txt` for a detailed list of pip package dependencies.

Tested on Ubuntu 22.04, CUDA 11.1 with NVIDIA A100 40GB and the following dependencies versions:
 * Python==3.8
 * PyTorch==1.9.1
 * segmentation_models_pytorch==0.3.2
 * timm==0.6.12
 * opencv-python==4.7.0.68
 * numpy==1.24.4

Recommended using Conda and installing dependencies as tested using: 

```bash
conda create -n=CeDiRNet-py3.8 python=3.8
conda activate CeDiRNet-py3.8

# install correct pytorch version for CUDA, e.g., for CUDA 11.1:
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

or directly from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

## Models

 * [MODEL](https://data.vicos.si/skokec/STEM/checkpoint.pth) trained for 1000 epochs with ConvNeXt-base on 512x512 STEM images using `src/config/nanoparticles/train.py` (SHA-256: `b77a30d6346309aeb64a7646d851db74d974758bf7d8e5f2cfcfd9f081637980`)
 * [MODALITY-DROPOUT MODEL](https://data.vicos.si/skokec/STEM/checkpoint-multimodality.pth) trained to stupport BF-only or HAADF-only image using the same 1000 epochs, ConvNeXt-base, 512x512, Adam learning rate `1e-4` and polynomial decay (exponent 2) recipe, (SHA-256: `6e515e20a6d4b83088b52bef507ee88b32504a0a321cbbf572f370b2d75956a1`)
 * pre-trained [LOCALIZATION](https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth) model for the second stage network trained on synthetic data only

The two final checkpoints were evaluated identically on 66 historical test image
pairs, disjoint from the 23 training pairs by image identity, using a fixed score
threshold of 0.5 and matching distance strictly below 20 pixels at 512x512:

| Test input | Original model F1 | Modality-dropout model F1 |
|---|---:|---:|
| BF + HAADF | 0.9579 | 0.9567 |
| BF-only | 0.7469 | 0.9498 |
| HAADF-only | 0.9474 | 0.9581 |

The observed paired-input penalty is **0.12 percentage points F1**, with gains of
**20.29 points for BF-only** and **1.07 points for HAADF-only**. These are single-run
comparisons on a reused historical holdout, not repeated-seed causal estimates.
Missing detectors were simulated by zeroing their channels; genuine singleton
acquisition shifts and missing-file support in loaders/serving APIs are not
validated by these results. Training dropout is disabled at inference.

## Usage

For training and evaluation on STEM dataset:

```bash
# download and extract dataset
dataset/download_dataset.sh

# run training and evaluation (on GPU by default!)
./run_nanoparticle_exp.sh
```

Script will print and display training progress (losses) as well as final results for all evaluated thresholds for the last epoch. Raw results for last epoch will be stored in JSON file in `./results/test_results` under folders for the corespoding thresholds.

The default configuration trains for 1000 epochs on 23 images of 512x512 pixels and evaluates checkpoints saved every 100 epochs. It requires less than 20GB of GPU memory on an NVIDIA A100 40GB. Inference time should be less than a few seconds per image on the same GPU.

Inference of images from any folder:
```bash
python infer.py --input_folder /path/to/images --img_pattern "*.png" --output_folder out/ --config src/config/config_infer.json --model path/to/checkpoint.pth

# Usage: infer.py [-h] [--input_folder INPUT_FOLDER] [--img_pattern IMG_PATTERN]
#                 [--output_folder OUTPUT_FOLDER] [--config CONFIG]
#                 [--model MODEL] [--localization_model LOCALIZATION_MODEL]
# 
# Process a folder of images with CeDiRNet.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --input_folder INPUT_FOLDER
#                         path to folder with input images
#   --img_pattern IMG_PATTERN
#                         pattern for input images
#   --output_folder OUTPUT_FOLDER
#                         path to output folder
#   --config CONFIG       path to config file
#   --model MODEL         path to model checkpoint file
#   --localization_model LOCALIZATION_MODEL
#                         (optional) path to localization model checkpoint file
#                         (will override one from model)
```

### Optional BF/HAADF modality dropout

To train a model that can also accept a missing detector, enable whole-channel
dropout in `src/config/nanoparticles/train.py`:

```python
modality_dropout=dict(bf_drop_probability=0.25, haadf_drop_probability=0.25),
```

Or use the existing config override mechanism (with the dataset environment
configured as for normal training):

```bash
DATASET=nanoparticles python src/train.py -c \
  modality_dropout.bf_drop_probability=0.25 \
  modality_dropout.haadf_drop_probability=0.25 \
  seed=17
```

The probabilities mean **which detector is removed**, not which is retained:

| Training input | Probability with the above settings |
|---|---:|
| Both BF and HAADF | 50% |
| BF-only (HAADF removed) | 25% |
| HAADF-only (BF removed) | 25% |

One categorical choice is drawn per sample; both detectors are never removed.
Probabilities must be finite, nonnegative and sum to at most one. Both default
to zero: old configs remain valid, and disabled dropout does not copy images or
consume random numbers. Enabling it does not change the architecture, losses,
labels or checkpoint tensor keys. The configuration is saved in `params.json`
and the policy is recorded under `modality_dropout` in training checkpoints.
Enabled dropout is incompatible with
`train_dataset.centerdir_gt_opts.use_cached_backbone_output=True`: cached outputs
bypass the image/backbone, so the Trainer rejects this combination before writing
run files or initializing data/models. Cached-output training remains allowed
when dropout is disabled.

Dropout runs **after dataset augmentation and before the existing FPN
normalization**, filling only the missing detector with raw zeros. Retained
channels are not rescaled. Inputs retain fixed slots `[BF, HAADF, auxiliary]`;
the normal auxiliary plane is zero, but existing augmentation of that plane is
left unchanged. Each epoch uses a private CPU generator seeded with
`seed + world_rank + epoch` (modulo `2**63 - 1`), independently of shuffle and
augmentation RNGs. This is epoch-level reproducibility, not exact mid-epoch resume.

#### Reuse in custom training loops / SLAIF Toolbox

`src/stem_modality.py` is host-independent and needs only PyTorch. Custom particle,
semantic or joint loops can call the same helper on floating CHW/BCHW tensors:

```python
from stem_modality import apply_modality_dropout, create_modality_dropout_generator

dropout = dict(bf_drop_probability=0.25, haadf_drop_probability=0.25)
# Once per epoch; reuse this generator across all batches in that epoch.
generator = create_modality_dropout_generator(seed=17, epoch=epoch)
# After the final input augmentation, before the model's normalization:
image = apply_modality_dropout(image, generator=generator, **dropout)
```

Apply it **once**, either in the native Trainer or in a custom loop, not both.
For deterministic evaluation, `apply_input_mode(image, mode)` supports `paired`,
`bf-only` and `haadf-only`, preserving the same detector slots. Training dropout is not applied
at inference. This option does not make the paired dataset loader accept missing
files, change serving APIs, or retrofit missing-modality robustness into old
weights; singleton file/annotation handling remains the adapter's responsibility.

Regression tests (install `pytest` in the model environment):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

### Adding new dataset

You may add new dataset by providing config files and dataset class:
 * add dataset class to `src/datasets/NEW_DATASET.py` and update `src/datasets/__init__.py` 
 * add `train.py` and `test.py` config files to `src/config/NEW_DATASET` and update `src/config/__init__.py`
