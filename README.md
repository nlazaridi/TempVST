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
