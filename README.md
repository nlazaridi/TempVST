# Temporal Visual Saliency Transformer (TempVST)

![avatar](https://github.com/nlazaridi/TempVST/blob/main/TempVST_arch.jpg)

## Installing the Conda Environment

To use this code, you need to install the required dependencies using Conda. Here's how to create a new Conda environment and install the dependencies:

1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the root directory of the cloned repository.
3. Create a new Conda environment using the following command:
```console
conda env create -f tempvst_env.yml
```
4. Activate the new Conda environment using the following command:
```console
conda activate tempvst_env
```

## Training, Testing, and Evaluation
Run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation. The predictions will be in `preds/` folder and the evaluation results will be in `result.txt` file.

### Testing on Our Pretrained TempVST Model
Run `python train_test_eval.py --Testing True --Evaluation True` for testing and evaluation. The predictions will be in `preds/` folder and the evaluation results will be in `result.txt` file.

# Script Arguments Explanation

This repository contains a script with various command-line arguments that control the behavior of the script. Below is an explanation of each argument and its purpose:

## Training and Testing Flags

- `--Training`: (default: False) Set this flag to True if you want to perform training.
- `--Testing`: (default: True) Set this flag to True if you want to perform testing.

## Learning Rate and Training Parameters

- `--lr_decay_gamma`: (default: 0.1) Learning rate decay factor.
- `--lr`: (default: 1e-4) Initial learning rate.
- `--epochs`: (default: 200) Number of training epochs.
- `--batch_size`: (default: 4) Batch size for training.
- `--num_gpu`: (default: 1) Number of GPUs to use.
- `--stepvalue1`: (default: 30000) First step value for adjusting the learning rate.
- `--stepvalue2`: (default: 45000) Second step value for adjusting the learning rate.
- `--trainset`: (default: 'DHF1K') Training dataset name.
- `--data_root`: Path to the data directory.
- `--img_size`: (default: 224) Size of network input images.
- `--alternate`: (default: 2) Subsampling factor.
- `--len_snippet`: (default: 6) Length of video snippet.
- `--pretrained_model`: (default: "80.7_T2T_ViT_t_14.pth.tar") Path to the pretrained model.

## Loss Function Coefficients

You can adjust the coefficients of various loss functions using the following arguments:

- `--kldiv_coeff`: (default: 1.0) Coefficient for KL Divergence loss.
- `--cc_coeff`: (default: -1.0) Coefficient for CC loss.
- `--sim_coeff`: (default: -1.0) Coefficient for Similarity loss.
- `--nss_coeff`: (default: 1.0) Coefficient for NSS loss.
- `--nss_emlnet_coeff`: (default: 1.0) Coefficient for NSS EMLNet loss.
- `--nss_norm_coeff`: (default: 1.0) Coefficient for NSS Normalization loss.
- `--l1_coeff`: (default: 1.0) Coefficient for L1 loss.

## Additional Flags

Various additional flags can be set to control the inclusion of specific components:

- `--kldiv`: (default: True) Set this flag to calculate KL Divergence.
- `--cc`: (default: False) Set this flag to include CC loss.
- `--sim`: (default: False) Set this flag to include Similarity loss.
- `--nss`: (default: False) Set this flag to include NSS loss.
- `--nss_emlnet`: (default: False) Set this flag to include NSS EMLNet loss.
- `--nss_norm`: (default: False) Set this flag to include NSS Normalization loss.
- `--l1`: (default: False) Set this flag to include L1 loss.

