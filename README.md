# Road-damage classifiers on RDD2022

Arya et al. (2024), *RDD2022: A multi-national image dataset for automatic road damage detection* (*Geoscience Data Journal*), release RDD2022 for deep convolutional detectors. The paper does not use fuzzy logic, ANFIS, or a genetic algorithm. This project adds that comparison on a small set of **annotated damage crops** from the Czech Republic subset.

It is not a new object detector and it does not claim to beat YOLO on the full 47,420-image set. Every model below sees the same four features and the same train/test crops. ANFIS is there because a neuro-fuzzy model can be read as rules. The MLP is the neural-network baseline. The two hybrids let a genetic algorithm search the nonlinear part while least squares sets the linear part.

## Dataset

Paper: [RDD2022: A multi-national image dataset for automatic road damage detection](https://doi.org/10.1002/gdj3.260) (Arya et al., Geoscience Data Journal, 2024).

Official dataset (full multi-country archive, about 13 GB):

https://doi.org/10.6084/m9.figshare.21431547.v1

Country packs, including the Czech zip (about 245 MB), are listed from the Sekilab repository:

https://github.com/sekilab/RoadDamageDetector

That Czech zip URL currently returns HTTP 403, so `prepare_crops.py --download` uses this public Hugging Face mirror of the same Czech training images:

https://huggingface.co/datasets/dronefreak/RDD2022

Those training labels contain longitudinal cracks, transverse cracks, and alligator cracks. They contain no pothole (D40) boxes, so the comparison uses the three classes that are present. Boxes in the mirror are YOLO labels:

| Code | Damage |
| --- | --- |
| D00 | longitudinal crack |
| D10 | transverse crack |
| D20 | alligator crack |
| D40 | pothole |

If you already have the official zip, put it at `data/RDD2022_Czech.zip` and run `prepare_crops.py` without `--download`. The script will read the PASCAL VOC XML annotations instead.

Only training images are used. The RDD2022 test images have no public boxes. Each kept box is cropped, then capped at 250 crops per class (or every box of a rarer class) and split 80/20 with the same class proportions for every model.

## Features

ANFIS cannot take raw pixels. Each crop becomes four inputs:

- bounding-box aspect ratio
- box area relative to the source image
- two principal components of a frozen MobileNetV2 embedding

The scaler and PCA are fit on the training crops only.

## Models

ANFIS is a first-order Takagi–Sugeno system written in `anfis.py`:

- 2 Gaussian membership functions per input (16 rules)
- product conjunction
- hybrid learning: ridge least squares for the rule consequents, then one Adam step on the membership centers and widths
- one regressor per class; the predicted class is the highest score

The baseline is a scikit-learn MLP `(32, 16)` trained on the identical matrix.

The genetic algorithm in `genetic.py` evolves a linear score for each class (bias plus one weight per feature). Fitness is training macro F1. It uses tournament selection, arithmetic crossover, Gaussian mutation, and elitism. The test crops are not part of the fitness.

`ga_anfis.py` joins those two. The chromosome is the shared Gaussian centres and widths (16 genes). For each chromosome, ridge least squares sets the rule consequents. Population 40, 40 generations, same operators, seed 42.

`ga_ann.py` joins the genetic algorithm with a neural network. The chromosome is the weights and biases of 8 tanh hidden units (40 genes). For each chromosome, ridge least squares sets one class score. Population 80, 80 generations, same operators, seed 42. The test crops are scored only after the search finishes.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Use the default PyTorch index instead of the CPU wheel if you want a CUDA build.

## Run

```bash
.venv/bin/python prepare_crops.py --download
.venv/bin/python train_eval.py
.venv/bin/python ga_anfis.py
.venv/bin/python ga_ann.py
```

`train_eval.py` writes the MLP, standalone ANFIS, and linear genetic algorithm:

- `outputs/metrics.json` — accuracy and per-class F1
- `outputs/classification_report.txt`
- `outputs/confusion_anfis.png`, `outputs/confusion_mlp.png`, `outputs/confusion_ga.png`
- `outputs/membership_functions.png`
- `outputs/rules.txt` — the strongest learned ANFIS rules for each class
- `outputs/ga_fitness.png`, `outputs/ga_weights.txt`
- `outputs/comparison_f1.png` and `outputs/comparison.txt` — all five models against the paper's detection score

`ga_anfis.py` writes:

- `outputs/ga_anfis.npz` — winning memberships and consequents
- `outputs/ga_anfis_metrics.json`, `outputs/ga_anfis_report.txt`, `outputs/ga_anfis_rules.txt`
- `outputs/ga_anfis_fitness.png`

`ga_ann.py` writes:

- `outputs/ga_ann.npz` — winning hidden layer and class scores
- `outputs/ga_ann_metrics.json`, `outputs/ga_ann_report.txt`, `outputs/ga_ann_network.txt`
- `outputs/ga_ann_fitness.png`, `outputs/confusion_ga_ann.png`

Score a saved hybrid again, without searching, with `ga_anfis.py --test` or `ga_ann.py --test`.

Diagrams in `diagrams/`:

- `overall_pipeline.png` — shared crops and features, then all five models
- `mlp_architecture.png` — the non-hybrid network
- `anfis_and_ga.png` — the two component models
- `ga_ann.png` — hidden layer evolved, class scores solved by least squares

`outputs/comparison_all.png` is the five-model F1 chart. The dashed line is the paper’s detection F1 of 0.770, which is a different task.

Low and High in the rules are relative to the standardized training distribution of each input, not to a physical unit.

## Result on this machine

Czech training crops, three classes (no D40 boxes in that subset): 501 train and 125 test.

| Model | Accuracy | Macro F1 | D00 | D10 | D20 |
| --- | --- | --- | --- | --- | --- |
| GA-ANN | 0.840 | 0.788 | 0.851 | 0.926 | 0.585 |
| Genetic algorithm | 0.832 | 0.780 | 0.854 | 0.914 | 0.571 |
| GA-ANFIS | 0.832 | 0.773 | 0.854 | 0.916 | 0.550 |
| MLP | 0.816 | 0.765 | 0.828 | 0.907 | 0.558 |
| ANFIS | 0.808 | 0.749 | 0.843 | 0.904 | 0.500 |

On this crop test GA-ANN is ahead. The linear genetic algorithm is next. GA-ANFIS is close to that linear model and is the hybrid that still writes fuzzy rules. Its strongest transverse-crack rule fires when the aspect ratio is High. The evolved linear scores agree: transverse cracks get a large positive aspect-ratio weight (+1.345), longitudinal cracks get a large negative one (−1.381), and alligator cracks get a positive area weight (+0.734). Several GA-ANN hidden units also put a weight larger than 1 on aspect ratio. The paper's winning detector is still a different task: F1 0.770 for finding damage in full images from six countries. See `outputs/comparison.txt`.
