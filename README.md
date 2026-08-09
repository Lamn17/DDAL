# ReCAL: Residual Coverage Active Learning

An implementation of **ReCAL** for pool-based active learning in YOLO object
detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> 🎯 **TL;DR**: ReCAL selects a diverse annotation batch by combining
> class-quality-aware detection uncertainty, residual coverage of the labeled
> set, and uncertainty-weighted clustering.

## 📖 About

ReCAL is an active-learning strategy for object detection. Each round trains a
YOLO detector on the labeled set, estimates a quality value for every class,
scores the current unlabeled pool, and selects `samples_per_round` images for
annotation.

### Key Features

- 🎯 **Class-aware uncertainty**: weighs per-box normalized posterior
  uncertainty by its predicted-class quality.
- 🔄 **Round-to-round quality tracking**: persists class-quality EMA values
  between active-learning rounds.
- 🧭 **Residual neighbour uncertainty**: emphasizes uncertain neighborhoods
  not already represented by labeled samples.
- 📐 **Uncertainty-weighted selection**: uses uncertainty-weighted K-means over
  detector features and returns one nearest-centre item per cluster.
- ⚙️ **YAML experiments**: provides ReCAL presets for VOC, KITTI, COCO,
  Cityscapes, GTSDB, and VTSDB100.
- 📊 **Selection-cost benchmark**: measures detector inference, feature
  extraction, scoring, and selection on a shared candidate pool.

This repository provides an implementation and experiment tooling; it does not
by itself establish a performance claim. Use matched seeds, initial labels,
budget, rounds, model, and candidate-pool policy when comparing strategies.

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.10 or higher (see `.python-version`)
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv)
- A YOLO-format detection dataset and its YAML file

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd FDAL-main
```

2. **Install dependencies with uv**:

```bash
# Install uv first if needed: https://docs.astral.sh/uv/
uv sync
```

3. **Set the process-title prefix required by the runner**:

```bash
export PROCTITLE_STARTSTR=fdal
```

Alternatively, add `PROCTITLE_STARTSTR=fdal` to `.env.training`; the runner
loads that file automatically.

### Prepare a Dataset

Set `dataset_yaml` in the chosen configuration to a valid YOLO dataset YAML.
Its `path` (or `root`) must resolve locally and its split entries must match
the dataset layout. The preset paths are local templates, not downloaded data.

### Run Your First ReCAL Experiment

```bash
# Run ReCAL on VOC
uv run python scripts/run_experiment.py \
  --config configs/voc/config_recal.yaml

```

## 🧠 ReCAL Method

### 1. Class-quality tracking during training

For every assigned foreground prediction, training records the quality signal

$$
q = p_{\mathrm{conf}}^{\xi}\operatorname{IoU}^{1-\xi}.
$$

The code maintains one exponential moving average (EMA) per class. The
previous round's `classwise_quality.npy` initializes the next round, so class
quality is carried through the active-learning cycle.

### 2. Difficulty-Calibrated Composite Uncertainty (DCCU)

For each predicted box $j$, the detector returns a class-probability vector.
Let $M_i$ be the number of retained boxes in image $i$, and let
$u_{ij}=H(p_{ij})(1-q_{\hat c_{ij}})$ be its class-quality-weighted
normalized posterior uncertainty. ReCAL aggregates the box scores as:

$$
U_i^{\mathrm{DCCU}} =
\begin{cases}
\sqrt{\dfrac{1}{M_i}\displaystyle\sum_{j=1}^{M_i} u_{ij}^{2}}, & M_i > 0, \\
0, & M_i = 0.
\end{cases}
$$

If class probabilities are unavailable, the strategy warns and falls back to
binary confidence uncertainty from detection confidence. The bundled YOLO
wrapper requests class probabilities and detector features, so normal ReCAL
runs use the posterior formulation.

### 3. Residual neighbour uncertainty (RNU)

RNU finds each candidate's `knn_k` nearest candidate features. It rewards an
uncertain neighbourhood only to the extent that its members are not already
covered by the labeled feature set. The final acquisition uncertainty is:

$$
u_i = u_i^{\mathrm{DCCU}} + u_i^{\mathrm{RNU}}.
$$

### 4. Uncertainty-weighted selection

ReCAL fits K-means with sample weights proportional to $u_i$, forms
`samples_per_round` clusters, and selects the feature nearest each cluster
centre. This yields one selection per cluster while biasing the partitioning
toward uncertain images.

## ⚙️ Configuration

Every dataset preset lives at `configs/<dataset>/config_recal.yaml`. Example:

```yaml
# configs/voc/config_recal.yaml
dataset_yaml: "datasets/VOC/data.yaml"
model_name: "yolo11s.pt"
strategy: "recal"
initial_labeled_count: 828
samples_per_round: 414
max_rounds: 7
num_inference: -1

strategy_args:
  recal:
    quality_momentum: 0.99
    quality_initial: 0.5
    quality_xi: 0.6
```

### Key Parameters

- `initial_labeled_count`: image count in the initial labeled set.
- `samples_per_round`: images selected for annotation after each round.
- `max_rounds`: number of active-learning rounds.
- `num_inference`: current unlabeled-pool images scored per round. `-1` means
  all current unlabeled images; a positive N scores the first N current
  unlabeled indices and must be at least `samples_per_round`.
- `knn_k`: candidate-neighbour count for RNU (default: 10).
- `sampling_conf`: detection confidence threshold for acquisition (default:
  0.25).
- `quality_momentum`, `quality_initial`, `quality_xi`: controls for the
  per-class EMA quality signal.

Included presets:

| Dataset | ReCAL configuration |
| --- | --- |
| VOC | `configs/voc/config_recal.yaml` |
| KITTI | `configs/kitti/config_recal.yaml` |
| COCO | `configs/coco/config_recal.yaml` |
| Cityscapes | `configs/cityscapes/config_recal.yaml` |
| GTSDB | `configs/gtsdb/config_recal.yaml` |
| VTSDB100 | `configs/vtsdb100/config_recal.yaml` |

## 🔧 Usage

### Custom Experiment

```bash
uv run python scripts/run_experiment.py \
  --config configs/voc/config_recal.yaml \
  --device 0 \
  --seed 10 \
  --num_inference 6000
```

### Resume an Experiment

```bash
uv run python scripts/run_experiment.py \
  --config configs/voc/config_recal.yaml \
  --resume_experiment_dir full_experiments/experiments_3010/voc_recal/<run> \
  --start_round 3
```

### Benchmark ReCAL Selection Cost

This command measures one controlled shared candidate pool. Its time includes
detector inference, feature extraction, scoring, and final selection; model
loading, warm-up, and artifact writes are excluded.

```bash
uv run python scripts/benchmark_selection.py \
  --configs \
    configs/voc/config_recal.yaml \
  --pool-size 6000 \
  --selection-budget 414 \
  --device 0 --warmup 2 --repeats 5
```

Results are written to `outputs/selection_benchmark/selection_benchmark.csv`
and `.md`. This benchmark is a controlled candidate-pool measurement, not a
full active-learning round.

## 📊 Outputs and Evaluation

Each run is saved below its `experiments_root`. The important outputs are:

- `round_<r>/train/.../weights/best.pt`: trained detector checkpoint.
- `round_<r>/train/.../weights/classwise_quality.npy`: per-class quality EMA.
- `round_<r>/metadata.yaml` and `selected_indices.npy`: round state and the
  selected images.
- `selection_log.txt`: selected image names, one row per query.
- `time_log.csv`: ReCAL acquisition time and processed-image count.

Evaluate active learning using detection metrics such as **mAP@0.5** and
**mAP@0.5:0.95**, and compare learning curves against total labeled budget.

## 📂 Project Structure

```text
FDAL-main/
├── configs/<dataset>/config_recal.yaml  # ReCAL experiment presets
├── scripts/
│   ├── run_experiment.py                # Active-learning runner
│   ├── train.py                         # YOLO training and quality tracking
│   ├── strategy.py                      # Strategy registry and execution
│   └── benchmark_selection.py           # Controlled selection-cost benchmark
├── src/
│   ├── models/yolo_model.py             # Features and class probabilities
│   └── strategies/intrinsic/recal.py    # ReCAL acquisition strategy
└── pyproject.toml                       # uv dependencies
```

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- Uses [scikit-learn](https://scikit-learn.org/) for nearest-neighbour search
  and weighted K-means.
# recal
