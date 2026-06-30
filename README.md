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

## Results
### WIth Incremental Learning Results

**Scene → Event**

| Method | | | |  | | |
|--------|--|--|--|--|--|--|
| | mAP | mIoU | F1 | F1@0.05 | F1@0.10 | Avg F1 (0.05~0.50) |
| EWC (PNAS'17) | 31.2 | 25.5 | 35.0 | 51.9 | 65.0 | 64.6 |
| BECAME (ICML'25) | 49.7 | 43.0 | 45.6 | <u>64.4</u> | **82.1** | 86.3 |
| PGM (ICML'25) | **55.6** | <u>45.5</u> | **54.6** | **64.6** | **82.1** | <u>86.5</u> |
| **Ours** | <u>53.4</u> | **47.1** | <u>52.6</u> | 64.2 | <u>81.9</u> | **86.7** |

**Event → Scene**

| Method | | | | | | |
|--------|--|--|--|--|--|--|
| | F1@0.05 | F1@0.10 | Avg F1 (0.05~0.50) | mAP | mIoU | F1 |
| EWC (PNAS'17) | 39.3 | 43.7 | 44.0 | 35.9 | 27.2 | 36.6 |
| BECAME (ICML'25) | 64.1 | 81.9 | 86.2 | 55.6 | 48.1 | 54.1 |
| PGM (ICML'25) | **65.8** | **82.5** | **86.9** | <u>56.0</u> | <u>49.0</u> | <u>54.2</u> |
| **Ours** | <u>64.7</u> | <u>82.1</u> | <u>86.5</u> | **57.4** | **50.2** | **55.4** |

### Only Encoder Results

| Method | | | | | | |
|--------|--|--|--|--|--|--|
| | mAP | mIoU | F1 | F1@0.05 | F1@0.10 | Avg F1 (0.05~0.50) |
| **Method originally designed for scene boundary detection** | | | | | | |
| LGSS‡ (CVPR'20) | 51.7 | 45.4 | 51.7 | 64.1 | 81.9 | 86.2 |
| CAT‡ (AAAI'23) | 52.7 | 46.0 | <u>52.0</u> | <u>64.6</u> | **82.4** | **86.7** |
| **Local-to-Global Context Encoder (Ours)** | **58.1** | **50.7** | **56.1** | 64.5 | 82.0 | 86.6 |
| **Method originally designed for event boundary detection** | | | | | | |
| PC‡ (ICCV'21) | 45.7 | 40.8 | <u>46.4</u> | 64.3 | 81.7 | 86.6 |
| Temporal Perceiver (TPAMI'23) | <u>53.3</u> | <u>53.2</u> | - | <u>74.8</u> | <u>82.8</u> | <u>86.0</u> |
| **Local-to-Global Context Encoder (Ours)** | **58.1** | **50.7** | **56.1** | 64.5 | 82.0 | 86.6 |

‡ indicates methods evaluated using the same feature inputs as ours.

### Ablation Study

Effect of each proposed module (LE = Local Encoder, GE = Global Encoder, D = Discriminator).

| LE | GE | D | Eval on Scene (S→E) | | | Eval on Event (S→E) | | | Eval on Event (E→S) | | | Eval on Scene (E→S) | | |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| | | | mAP | mIoU | F1 | F1@0.05 | F1@0.10 | Avg F1 | F1@0.05 | F1@0.10 | Avg F1 | mAP | mIoU | F1 |
| ✓ | ✗ | ✗ | 19.3 | 4.8 | 24.6 | 64.4 | 82.0 | 86.4 | 64.2 | 82.1 | 86.4 | 26.8 | 22.3 | 29.5 |
| ✓ | ✓ | ✗ | 28.5 | 19.4 | 43.6 | 64.4 | 81.3 | 85.7 | 56.7 | 76.9 | 82.0 | 54.5 | 47.4 | 53.8 |
| ✓ | ✓ | ✓ | **53.4** | **47.1** | **52.6** | **64.2** | **81.9** | **86.7** | **64.7** | **82.1** | **86.5** | **57.4** | **50.2** | **55.4** |
## Citation
```
@inproceedings{tian2026temporal,
  title     = {Multi-Type Incremental Local-to-Global Context Encoder for Video Boundary Detection},
  author    = {Xiaoxuan Tian, Ruifan Zhao, Zhilong Ou, and Hongxing Wang},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```
