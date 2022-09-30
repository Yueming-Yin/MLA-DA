# MLA-DA: Metric-Learning-Assisted Domain Adaptation
Code release for **[Metric-Learning-Assisted Domain Adaptation](https://www.sciencedirect.com/science/article/pii/S0925231221007608)** (Neurocomputing, 2021, IF: 5.779)

- All rights reserved by Yueming Yin, Email: 1018010514@njupt.edu.cn (or yinym96@qq.com).

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

## Citation
Please cite:
```
@article{yin2021metric,
  title={Metric-learning-assisted domain adaptation},
  author={Yin, Yueming and Yang, Zhen and Hu, Haifeng and Wu, Xiaofu},
  journal={Neurocomputing},
  volume={454},
  pages={268--279},
  year={2021},
  publisher={Elsevier}
}
```
or
```
Yin, Yueming, Zhen Yang, Haifeng Hu, and Xiaofu Wu. "Metric-learning-assisted domain adaptation." Neurocomputing 454 (2021): 268-279.
```

## Contact
- 1018010514@njupt.edu.cn (Yueming Yin)
