# DDAL: Difficulty Distribution Active Learning

An implementation of **Difficulty Distribution Active Learning (DDAL)** for
pool-based active learning in YOLO object detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> 🎯 **TL;DR**: DDAL separates classification and localization difficulty,
> estimates the class-aspect yield of each image, and optimizes explicit quota
> targets with Budgeted Class Query Selection.

## 📖 About

Difficulty Distribution Active Learning (DDAL) is an active-learning strategy
for object detection. Each round trains a YOLO detector on the labeled set,
estimates classification and localization quality for every class, and selects
`samples_per_round` images for annotation.

### Key Features

- 🎯 **Dual-aspect quality**: keeps classification confidence and localization
  IoU in separate per-class EMA vectors.
- 📦 **Aspect-matched yield**: combines class posterior entropy with DFL
  localization entropy without collapsing detections into one image score.
- 📐 **Quota saturation**: allocates a target to every class-aspect pair and
  selects with deterministic submodular greedy.
- ⚙️ **YAML experiments**: provides DDAL presets for VOC, KITTI

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
git clone https://github.com/Lamn17/ddal.git
cd ddal
```

2. **Install dependencies with uv**:

```bash
# Install uv first if needed: https://docs.astral.sh/uv/
uv sync
```

3. **Set the process-title prefix required by the runner**:

```bash
export PROCTITLE_STARTSTR=ddal
```

Alternatively, add `PROCTITLE_STARTSTR=ddal` to `.env.training`; the runner
loads that file automatically.

### Prepare a Dataset

Set `dataset_yaml` in the chosen configuration to a valid YOLO dataset YAML.
Its `path` (or `root`) must resolve locally and its split entries must match
the dataset layout. The preset paths are local templates, not downloaded data.

### Run Your First DDAL Experiment

```bash
# Run DDAL on VOC
uv run python scripts/run_experiment.py \
  --config configs/voc/config_ddal.yaml
```

## 🧠 DDAL Method

### 1. Difficulty Class Quality Estimation (DCQE)

For every assigned foreground prediction, training records two signals:

$$
q_n^{\mathrm{cls}}=p_{n,c}^{\mathrm{conf}},\qquad
q_n^{\mathrm{loc}}=\mathrm{IoU}(\hat b_n,b_n).
$$

The code maintains a separate EMA for each class and aspect. Difficulty is:

$$
D_c^{\mathrm{cls}}=1-Q_c^{\mathrm{cls}},\qquad
D_c^{\mathrm{loc}}=1-Q_c^{\mathrm{loc}}.
$$

### 2. Dual Entropy Class Scoring (DECS)

For predicted box $j$ in image $i$, classification uncertainty is normalized
posterior entropy:

$$
H_{ij}=-\frac{\sum_c p_{ijc}\log(p_{ijc}+\epsilon)}{\log C}.
$$

Localization uncertainty is the mean normalized entropy of the four DFL side
distributions (with `reg_max=16` for YOLO11):

$$
V_{ij}=\frac14\sum_{s\in\{l,t,r,b\}}
-\frac{\sum_{k=1}^{16}d_{ij}^{s,k}\log(d_{ij}^{s,k}+\epsilon)}{\log16}.
$$

If raw DFL logits are unavailable, $V_{ij}=1-p_{ij}^{\mathrm{conf}}$. The image
retains a yield for every class and aspect:

$$
\hat y_{ic}^{\mathrm{cls}}=\sum_jp_{ijc}H_{ij},\qquad
\hat y_{ic}^{\mathrm{loc}}=\sum_jp_{ijc}V_{ij}.
$$

### 3. Budgeted Class Query Selection (BCQS)

Let $\bar m_L$ be the mean labeled object count per image. DDAL allocates:

$$
\bar B=B\bar m_L,\qquad
b_c^a=\frac{\bar B D_c^a}{\sum_{c'}\sum_{a'}D_{c'}^{a'}}.
$$

It then maximizes:

$$
F(S)=\sum_c\sum_{a\in\{\mathrm{cls},\mathrm{loc}\}}
\min\left(\sum_{i\in S}\hat y_{ic}^a,b_c^a\right).
$$

The budgeted class-query implementation uses deterministic saturation greedy and has the standard
$(1-1/e)$ guarantee for this monotone submodular objective. The complete
formulas and pseudocode are described above.

## ⚙️ Configuration

Every dataset preset lives at `configs/<dataset>/config_ddal.yaml`. Example:

```yaml
# configs/voc/config_ddal.yaml
dataset_yaml: "datasets/VOC/data.yaml"
model_name: "yolo11s.pt"
strategy: "ddal"
initial_labeled_count: 828
samples_per_round: 414
max_rounds: 7
num_inference: -1

strategy_args:
  ddal:
    sampling_conf: 0.25
    quality_momentum: 0.99
    quality_initial: 0.5
    difficulty_mode: "dual"
    signal_mode: "both"
    selection_mode: "bcqs"
```

### Key Parameters

- `initial_labeled_count`: image count in the initial labeled set.
- `samples_per_round`: images selected for annotation after each round.
- `max_rounds`: number of active-learning rounds.
- `num_inference`: current unlabeled-pool images scored per round. `-1` means
  all current unlabeled images; a positive N scores the first N current
  unlabeled indices and must be at least `samples_per_round`.
- `sampling_conf`: detection confidence threshold for acquisition (default:
  0.25).
- `quality_momentum`, `quality_initial`: control the two per-class quality EMA
  vectors. DDAL no longer uses `quality_xi`.
- `difficulty_mode`: `dual` for full DCQE, `uniform` for equal quotas, or
  `combined` for the paired difficulty $1-Q_c^{cls}Q_c^{loc}$.
- `signal_mode`: `both`, `cls`, or `loc` controls which DECS channel receives
  quota and contributes to selection.
- `selection_mode`: `bcqs` uses quota saturation; `topk` selects by the raw
  sum of active class-aspect scores.

Included presets:

| Dataset | DDAL configuration |
| --- | --- |
| VOC | `configs/voc/config_ddal.yaml` |
| KITTI | `configs/kitti/config_ddal.yaml` |

## 🔧 Usage

### Custom Experiment

```bash
uv run python scripts/run_experiment.py \
  --config configs/voc/config_ddal.yaml \
  --device 0 \
  --seed 10 \
  --num_inference -1
```

### Resume an Experiment

```bash
uv run python scripts/run_experiment.py \
  --config configs/voc/config_ddal.yaml \
  --resume_experiment_dir full_experiments/experiments_3010/voc_ddal/<run> \
  --start_round 3
```

## 📊 Outputs and Evaluation

Each run is saved below its `experiments_root`. The important outputs are:

- `round_<r>/train/.../weights/best.pt`: trained detector checkpoint.
- `round_<r>/train/.../weights/classwise_quality_cls.npy`: classification-quality EMA.
- `round_<r>/train/.../weights/classwise_quality_loc.npy`: localization-quality EMA.
- `round_<r>/metadata.yaml` and `selected_indices.npy`: round state and the
  selected images.
- `selection_log.txt`: selected image names, one row per query.
- `time_log.csv`: DDAL acquisition time and processed-image count.
- `ddal_state.npz`: dual quality, difficulty, quota, and predicted-yield state.
- `ddal_aspect_dynamics.csv`: per-round $Q_c^a$ and $D_c^a$ values.
- `ddal_allocation_fidelity.csv`: quota, predicted yield, and selected GT count.
- `ddal_bcqs_selection_metrics.csv`: BCQS timing, quota coverage, and DFL statistics.

Evaluate active learning using detection metrics such as **mAP@0.5** and
**mAP@0.5:0.95**, and compare learning curves against total labeled budget.

## 📂 Project Structure

```text
DDAL/
├── configs/<dataset>/config_ddal.yaml  # DDAL experiment presets
├── scripts/
│   ├── run_experiment.py                # Active-learning runner
│   ├── setup_data.py                    # Initial labeled-pool setup
│   ├── train.py                         # YOLO training and quality tracking
│   ├── strategy.py                      # Strategy registry and execution
│   ├── simulate_labeling.py             # Round-to-round label simulation
│   └── utils.py                         # Shared model and experiment helpers
├── src/
│   ├── models/yolo_model.py             # Class posteriors and DFL uncertainty
│   └── strategies/intrinsic/ddal.py     # DDAL acquisition strategy
└── pyproject.toml                       # uv dependencies
```

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
