Semi-supervised Learning Multiclass Anomaly Detection in Multivariate Timeseries
==============

**Author:** *[Wooheon Hong](https://www.linkedin.com/in/wooheon-hong-b33621200/)*, *[Minsoo Kim](https://github.com/km19809)* with Samsung Electronics

**Date:** 2021.05.10 ~ 2022. 04. 29

This is the pytorch implementation of MAD (Multiclass Anomaly Detection)

# MPUMAD 

Multi-class PUMAD(MPUMAD)

1. Pseudo labeling with multi-class DHF (MDHF)
Perform the DHF for each anomaly class
Find reliable normal as an intersection of DHF

2. Deep metric learning with quadruple loss
Use quadruple loss to distinguish between anomaly classes in metric learning
(Triple loss tends to pull anomaly classes together in metric learning)

3. Repeat 1 and 2 

4. Classify by the distance between normal centroid and each anomaly centroid




# Train & Evaluation 


### CE  
```
python main.py {data_name} {data_path} {output_dir} --window 10 --normal_class 0 --is_ood --labeled_normal_ratio 0.1 --labeled_anomaly_ratio 0.1 --model_name base_model --net_name wavenet --lr 0.0001 --n_epochs 150 --hidden_channels 128 --batch_size 128 
```

### PseudoLabeling + ABC
```
python main.py {data_name} {data_path} {output_dir} --window 10 --normal_class 0 --is_ood --labeled_normal_ratio 0.1 --labeled_anomaly_ratio 0.1 --model_name PseudoLabeling --net_name wavenet --lr 0.0001 --n_epochs 150 --hidden_channels 128 --batch_size 128 --class_imbalance_name abc  
```

The detailed descriptions about the parameters are as following:

| Parameter name        | Description of parameter                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| data_name             | File name of input .npy                                                                                                   |
| data_path             | Directory location with the train data and test data                                                                      |
| output_dir            | Directory location of the results, default log/                                                                           |
| window                | Length of sliding window, default 10                                                                                      |
| normal_class          | Set which class is the normal class of the dataset (all other classes except ood class are considered anomaly), default 0 |
| is_ood                | Decide whether to experiment or not in the out-of-distribution setting                                                    |
| labeled_normal_ratio  | Ratio of labeled normal training examples, default 0.01                                                                   |
| labeled_anomaly_ratio | Ratio of labeled anomaly training examples, default 0.01                                                                  |
| load_config           | Config JSON-file path, default None                                                                                       |
| load_model            | Saved model file path, default None                                                                                       |
| model_name            | Set which model to use                                                                                                    |
| margin_name           | Set which margin to use or not, default None                                                                              |
| net_name              | Set which neural network to use for embedding                                                                             |
| class_imbalance_name  | Set which class imbalance loss to use or not, default None                                                                |
| eta                   | ..                                                                                                                        |
| hidden_channels       | Dimension size for latent representation                                                                                  |
| seed                  | Set seed.                                                                                                                 |
| n_epochs              | Epoch size during training, default 100                                                                                   |
| lr                    | Initial learning rate of Adam optimizer, default 0.0001                                                                   |
| lr_milestone          | Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.                         |
| weight_decay          | Weight decay (L2 penalty) hyperparameter for ..                                                                           |
| batch_size            | Batch size for mini-batch training, default 128                                                                           |
| device                | Computation device to use ("cpu", "cuda", "cuda:2", etc.), default "cuda"                                                 |
| num_threads           | Number of threads used for parallelizing CPU operations. 0 means that all resources are used, default 0                   |
| n_jobs_dataloader     | Number of workers for data loading. 0 means that the data will be loaded in the main process, default 0                   |

# Models

The detailed descriptions about the models are as following:

| Model name     | Description of model          |
| -------------- | ----------------------------- |
| base_model     | Cross entropy                 |
| DeepMCDD       | DeepMCDD, OOD detection model |
| PseudoLabeling | PseudoLabeling, SSL model     |
| MPUMAD         | MPUMAD, MPU model             |

# Code Structure 

```
├── base
│   ├── base_dataset.py
│   ├── base_net.py
│   ├── base_trainer.py
│   ├── __init__.py
│   └── torchtimeseries_dataset.py
├── datasets
│   ├── __init__.py
│   ├── main.py
│   ├── nsl_kdd.py
│   ├── preprocessing.py
│   └── unsw_nb15.py
├── log
│   ├── config.json
│   ├── model.tar
│   └── results.json
├── mad.py
├── main.py
├── models
│   ├── base_model.py
│   ├── deepmcdd.py
│   ├── embedding
│   │   ├── mlp.py
│   │   └── network_data_wavenet.py
│   ├── __init__.py
│   ├── main.py
│   ├── mb.py
│   └── mdeepsad.py
├── optim
│   ├── deepmcdd_abc_trainer.py
│   ├── deepmcdd_trainer.py
│   ├── face_trainer.py
│   ├── __init__.py
│   ├── main.py
│   ├── margin
│   │   ├── ArcMarginProductAdaptive.py
│   │   ├── ArcMarginProduct.py
│   │   ├── CosineMarginProduct.py
│   │   ├── __init__.py
│   │   └── MultiMarginProduct.py
│   ├── mb_trainer.py
│   ├── mdeepsad_trainer.py
│   ├── pseudolabeling_abc_trainer.py
│   ├── pseudolabeling_trainer.py
│   ├── semi_deepmcdd_trainer.py
│   ├── supervised_abc_trainer.py
│   ├── supervised_focal_trainer.py
│   └── supervised_trainer.py
├── README.md
└── utils
    ├── config.py
    ├── focal.py
    ├── __init__.py
    ├── loss.py
    ├── utils_mpumad.py
    └── utils.py
```
