# MLA-DA: Metric-Learning-Assisted Domain Adaptation
Code release for Metric-Learning-Assisted Domain Adaptation (arxiv: 2004.10963)
## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- download datasets

- modify your root path, task name and GPU ids

    Line 38-41 in train.py

- train:

  `python train.py`
  
- monitor (tensorboard required)

  `tensorboard --logdir .\log --port 8888`
  
- view the results of our ablation study

  `tensorboard --logdir .\Ablation Study --port 8888`
