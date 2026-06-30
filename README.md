# Multi-Type Incremental Local-to-Global Context Encoder for Video Boundary Detection

## Environment

This project runs on Linux (Ubuntu) with 4090 GPU.

Install the following packages at first:

* python 3.10.x
* pytorch 2.1.1+cu121
* torchvision 0.16.1+cu121
* torchmetric 1.4.0.post0

Then use the following command to install dependencies.
`pip install -r requirements.txt`

## Data Preparation

Please follow the instructions in the repositories below to prepare the datasets:

**Movienet**：[https://github.com/mini-mind/VSMBD](https://github.com/mini-mind/VSMBD)

then you should obtain the following files:

```
ImageNet_shot.pkl
Places_shot.pkl
```

**Kinetics-GEBD**:[https://github.com/MCG-NJU/TemporalPerceiver](https://github.com/MCG-NJU/TemporalPerceiver)

then you should obtain the following files:

```
features
k400_mr345_train_min_change_duration0.3.pkl
k400_mr345_val_min_change_duration0.3.pkl
```

After preparation, place the data under the `data/` directory (or update the path in the config file accordingly).

## Usage

**our method**
train without incremental learning
`python event_train.py`
`python scene_train.py`

train with incremental learning
`python event2sceneWithCL.py`
`python scene2eventWithCL.py`

eval
`python event_eval.py`
`python scene_eval.py`

**EWC method**
This implement is built upon [EWC](https://github.com/ariseff/overcoming-catastrophic). Thanks for their great work.
train with EWC method
`python event_train.py`
`python scene_train.py`

**BECAME method**
This implement is built upon [BECAME](https://github.com/limei0818/BECAME). Thanks for their great work.
train with BECAME method
`python event_train.py`
`python scene_train.py`

**PGM method**
This implement is built upon [PGM](https://github.com/njustkmg/ICML25-PGM). Thanks for their great work.
train with PGM method
`python event_train.py`
`python scene_train.py`

**LGSS method**
This implement is built upon [LGSS](https://github.com/anyirao/SceneSeg). Thanks for their great work.
train with LGSS method
`python event_train.py`
`python scene_train.py`

**CAT method**
This implement is built upon [CAT](https://github.com/njustkmg/AAAI23_CAT). Thanks for their great work.
train with CAT method
`python event_train.py`
`python scene_train.py`

**PC method**
This implement is built upon [PC](https://github.com/StanLei52/GEBD). Thanks for their great work.
train with PC method
`python event_train.py`
`python scene_train.py`
