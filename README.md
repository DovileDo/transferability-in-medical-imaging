# Dataset transferability in medical image classification

This repository contains the code and results from the paper [*On dataset transferability in medical image classification*](https://arxiv.org/pdf/2412.20172). The study introduces a novel transferability metric that integrates feature quality and gradients to assess the suitability and adaptability of source model features for various target tasks. 

This PyTorch implementation enables the calculation of transferability scores and provides pre-training and fine-tuning scripts for medical image classification tasks, using datasets from [MedMNIST](https://medmnist.com/).

## Features

- Implements a novel transferability metric combining feature quality and gradients.
- Supports comparison with existing metrics such as LEEP, NLEEP, and others.
- Provides tools for leave-target-out pretraining on MedMNIST, ResNet18 pretraining on [RadImageNet](https://github.com/BMEII-AI/RadImageNet), and fine-tuning models on target tasks from MedMNIST. 
- Designed specifically for medical imaging datasets.

For access to the models and data used in the paper, please refer to:
* [MedMNIST datasets and pre-trained models](https://medmnist.com/) (models available here: [https://zenodo.org/records/7782114?preview_file=weights_chestmnist.zip](https://zenodo.org/records/7782114) we used resnet18_224_1),
* [RadImageNet](https://github.com/BMEII-AI/RadImageNet),
* [Leave-target-out MedMNIST pretraining](https://osf.io/4zgrd/), and
* [Subsets of MedMNIST datasets used in our paper](https://osf.io/4zgrd/).

## Setup

### Requirements
Dependencies are provided in the `conda.yaml` file. To set up the environment:

```bash
conda env create -f conda.yaml
conda activate your-environment-name
```

## Usage

### Fine-Tuning

To fine-tune a source model for a target task, use the following command:

```bash
python -m fine_tuning.py --source_flag <MedMNIST source flag> --target_flag <MedMNIST target flag> --lr <learning rate> --epochs <number of epochs> --batch_size <batch size>
```
Replace `<MedMNIST source flag>` and `<MedMNIST target flag>` with the appropriate dataset flags from MedMNIST. Adjust learning rate, epochs, and batch size as needed.

### Calculating Transferability Scores

To compute transferability scores for dataset or model evaluation, run:
```bash
python -m transferability_scores.py --source <dataset or model> --method <FU | LP | LEEP | NLEEP | others>
```
`--source`: Specify `dataset` for dataset transferability or `model` for architecture transferability.
* Source datasets: 12 datasets from MedMNIST, leave-target-out MedMNIST, RadImageNet, ImageNet
* Implemented architectures: densenet, efficientnet, googlenet, mnasnet, mobilenet, vgg, convnext, shufflenet, resnet
  
`--method`: Choose a transferability metric. Options include:
* FU or LP proposed in the paper,
* LEEP, NLEEP, LogME, PARC, SFDA, NCTI.

### Results

The results from the experiments, including transferability scores and fine-tuning performance, are available in the `results` folder and are analyzed in the `dataset_transferability`, `architecture_transferability`, and `finetuned_AUCs` notebooks.

## Contact

Feel free to contact us for help with reproducing our experiments or if you have any questions about this repository.

## Citation
If you find our method useful in your research, please cite:

```yaml
@article{juodelyte2024dataset,
  title={On dataset transferability in medical image classification},
  author={Juodelyte, Dovile and Ferrante, Enzo and Lu, Yucheng and Singh, Prabhant and Vanschoren, Joaquin and Cheplygina, Veronika},
  journal={arXiv preprint arXiv:2412.20172},
  year={2024}
}
```

