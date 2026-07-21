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
 * pre-trained [LOCALIZATION](https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth) model for the second stage network trained on synthetic data only

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

### Adding new dataset

You may add new dataset by providing config files and dataset class:
 * add dataset class to `src/datasets/NEW_DATASET.py` and update `src/datasets/__init__.py` 
 * add `train.py` and `test.py` config files to `src/config/NEW_DATASET` and update `src/config/__init__.py`
