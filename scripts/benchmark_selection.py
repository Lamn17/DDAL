"""Benchmark end-to-end active-learning selection cost on a real image pool.

The timed region is ``strategy.query()``: detector inference, feature
extraction, acquisition scoring and sample selection.  Model loading, warm-up
iterations and optional THOP complexity profiling are deliberately excluded.
Prediction dumps and selection symlinks are disabled so storage I/O does not
dominate the measurement.
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from thop import profile

import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.utils import create_model


@dataclass(frozen=True)
class DatasetPool:
    image_paths: list[str]
    train_indices: np.ndarray
    class_names: list[str]


STRATEGY_CLASSES: dict[str, tuple[str, str]] = {
    "random": ("src.strategies.random.random_strategy", "RandomStrategy"),
    "entropy": ("src.strategies.uncertainty.entropy", "EntropyStrategy"),
    "margin": ("src.strategies.uncertainty.margin", "MarginStrategy"),
    "coreset": ("src.strategies.diversity.coreset", "CoreSetStrategy"),
    "badge": ("src.strategies.intrinsic.badge", "BADGEStrategy"),
    "fdal": ("src.strategies.uncertainty.fdal", "FDAL"),
    "ccms": ("src.strategies.diversity.ccms", "CCMSStrategy"),
    "dcus": ("src.strategies.uncertainty.dcus", "DCUSStrategy"),
    "ddus": ("src.strategies.uncertainty.ddus", "DDUSStrategy"),
    "maple": ("src.strategies.uncertainty.maple_uncertainty", "MaPLeUncertaintyStrategy"),
    "cdal": ("src.strategies.intrinsic.cdal", "CDALStrategy"),
    "divproto": ("src.strategies.intrinsic.divproto", "DivProtoStrategy"),
    "midprc": ("src.strategies.intrinsic.midprc", "MIDPRCStrategy"),
    "recal": ("src.strategies.intrinsic.recal", "ReCALStrategy"),
}


def create_strategy(strategy_name: str, model: Any, **kwargs: Any) -> Any:
    """Import only the requested strategy, avoiding optional-method imports."""
    try:
        module_name, class_name = STRATEGY_CLASSES[strategy_name]
    except KeyError as exc:
        supported = ", ".join(sorted(STRATEGY_CLASSES))
        raise ValueError(f"Unknown strategy '{strategy_name}'. Supported: {supported}") from exc
    return getattr(import_module(module_name), class_name)(model, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end active-learning selection time on real images."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Experiment YAML files to compare. They should point to the same dataset.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1000,
        help="Unlabeled candidate images timed per method (default: 1000).",
    )
    parser.add_argument(
        "--selection-budget",
        type=int,
        default=None,
        help="Images selected per query; defaults to samples_per_round in each config.",
    )
    parser.add_argument(
        "--labeled-size",
        type=int,
        default=None,
        help="Fixed labeled set size; defaults to initial_labeled_count in each config.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Untimed query runs per method.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed query runs per method.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device override, for example 0, 1, cpu or auto (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional detector checkpoint/name override applied to every config.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default=None,
        help="Optional dataset YAML override applied to every config; useful for Kaggle paths.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used to choose the shared labeled/pool split."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/selection_benchmark",
        help="Directory for selection_benchmark.csv and selection_benchmark.md.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip parameter and MAC/FLOP profiling (useful when only timing is needed).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input resolution for FLOP profiling; defaults to imgsz in each config.",
    )
    args = parser.parse_args()
    if args.pool_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("pool-size and repeats must be positive; warmup cannot be negative")
    if args.selection_budget is not None and args.selection_budget <= 0:
        parser.error("selection-budget must be positive")
    if args.labeled_size is not None and args.labeled_size < 0:
        parser.error("labeled-size cannot be negative")
    return args


def read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return value


def _class_names(data_config: dict[str, Any]) -> list[str]:
    names = data_config.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda key: int(key))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("Dataset YAML needs a list or mapping in 'names'.")


def load_dataset_pool(config: dict[str, Any]) -> DatasetPool:
    # Import lazily: importing ``src`` also imports the YOLO wrapper, while
    # ``--help`` and config validation should not initialize model code.
    from src.data.dataset import ALDataset

    dataset_yaml = config.get("dataset_yaml")
    if not dataset_yaml:
        raise ValueError("Config is missing dataset_yaml")
    data_config = read_yaml(dataset_yaml)
    dataset_root = data_config.get("path") or data_config.get("root")
    if not dataset_root:
        raise ValueError(f"Dataset YAML is missing path/root: {dataset_yaml}")
    dataset = ALDataset(str(dataset_root), Path(str(dataset_yaml)).stem, _class_names(data_config))
    return DatasetPool(
        image_paths=dataset.get_image_paths(),
        train_indices=np.asarray(dataset.train_indices, dtype=int),
        class_names=dataset.class_names,
    )


def shared_split(
    pool: DatasetPool, labeled_size: int, pool_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if labeled_size + pool_size > len(pool.train_indices):
        raise ValueError(
            f"Need {labeled_size + pool_size} training images but dataset has only "
            f"{len(pool.train_indices)}. Reduce --pool-size or --labeled-size."
        )
    permutation = np.random.default_rng(seed).permutation(pool.train_indices)
    return permutation[:labeled_size], permutation[labeled_size:labeled_size + pool_size]


def _strategy_names(config: dict[str, Any]) -> list[str]:
    configured = config.get("strategy")
    if isinstance(configured, str):
        return [name.strip() for name in configured.split("-") if name.strip()]
    if isinstance(configured, list) and all(isinstance(name, str) for name in configured):
        return list(configured)
    raise ValueError("Config 'strategy' must be a string or a list of strings")


def _synchronise_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _silence_artifact_writes(strategy: Any) -> None:
    # Strategy implementations use these helpers after the selection algorithm.
    # They create thousands of files/symlinks and are not acquisition cost.
    strategy._save_predictions_for_selection = lambda **_: None
    strategy._save_selection_symlinks = lambda *_args, **_kwargs: None


def _strategy_kwargs(
    config: dict[str, Any], strategy_name: str, experiment_dir: str,
    labeled_indices: np.ndarray, device: str,
) -> dict[str, Any]:
    configured_device = config.get("device", "auto") if device == "auto" else device
    kwargs: dict[str, Any] = {
        "experiment_dir": experiment_dir,
        "round": 1,
        "device": configured_device,
        "inference_device": configured_device,
        "inference_batch_size": config.get("inference_batch_size", 1),
        "train_ddp": config.get("train_ddp", False),
        "master_addr": config.get("master_addr", "localhost"),
        "master_port": config.get("master_port", 12355),
        "backend": config.get("backend", "nccl"),
        "seed": config.get("seed", 42),
        "data_yaml": config.get("dataset_yaml"),
        "labeled_indices": labeled_indices,
    }
    strategy_args = config.get("strategy_args", {})
    if not isinstance(strategy_args, dict):
        raise ValueError("strategy_args must be a mapping")
    extra_args = strategy_args.get(strategy_name, {})
    if not isinstance(extra_args, dict):
        raise ValueError(f"strategy_args.{strategy_name} must be a mapping")
    kwargs.update(extra_args)
    return kwargs


def run_query(
    config: dict[str, Any], model: Any, candidate_indices: np.ndarray,
    pool: DatasetPool, labeled_indices: np.ndarray, budget: int, device: str,
    experiment_dir: str,
) -> np.ndarray:
    """Run the config's single or chained strategy once, without result artifacts."""
    strategy_names = _strategy_names(config)
    expand_ratios = config.get("expand_ratios", [])
    if len(strategy_names) > 1 and len(expand_ratios) != len(strategy_names) - 1:
        raise ValueError("A chained strategy needs exactly one expand ratio per non-final stage")

    current_indices = candidate_indices.copy()
    for stage, strategy_name in enumerate(strategy_names):
        stage_budget = budget
        if stage < len(strategy_names) - 1:
            stage_budget = int(budget * np.prod(expand_ratios[stage:]))
        stage_budget = min(stage_budget, len(current_indices))
        strategy = create_strategy(
            strategy_name,
            model,
            **_strategy_kwargs(config, strategy_name, experiment_dir, labeled_indices, device),
        )
        _silence_artifact_writes(strategy)
        current_indices = np.asarray(
            strategy.query(
                unlabeled_indices=current_indices,
                image_paths=pool.image_paths,
                n_samples=stage_budget,
                num_inference=len(current_indices),
            ),
            dtype=int,
        )
    return current_indices


def profile_detector(model: Any, input_size: int) -> tuple[int, float, float]:
    """Return params, GMACs, and conventional GFLOPs (two FLOPs per MAC)."""
    detector = model.model.model
    detector.eval()
    parameters = sum(parameter.numel() for parameter in detector.parameters())
    dummy_input = torch.randn(1, 3, input_size, input_size)
    macs, _ = profile(detector, inputs=(dummy_input,), verbose=False)
    return parameters, macs / 1e9, (2.0 * macs) / 1e9


def profile_extra_models(config: dict[str, Any], class_count: int) -> tuple[int, float, float, str]:
    """Profile extra inference modules currently constructed by a strategy."""
    if "fdal" not in _strategy_names(config):
        return 0, 0.0, 0.0, ""
    from src.strategies.uncertainty.fdal import ResNetClassifier

    args = config.get("strategy_args", {}).get("fdal", {})
    supporter = ResNetClassifier(
        arch_name=args.get("supporter", "resnet18"),
        n_label=class_count,
        pretrained=False,
        emb_size=args.get("supporter_embedding_size", 256),
        dropout=args.get("supporter_dropout", 0.2),
        fine_tune_layers=args.get("fine_tune_layers", -1),
    )
    supporter.eval()
    input_size = int(args.get("supporter_imgsz", 320))
    parameters = sum(parameter.numel() for parameter in supporter.parameters())
    macs, _ = profile(supporter, inputs=(torch.randn(1, 3, input_size, input_size),), verbose=False)
    return parameters, macs / 1e9, (2.0 * macs) / 1e9, type(supporter).__name__


def population_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selection_benchmark.csv"
    columns = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "selection_benchmark.md"
    with markdown_path.open("w") as handle:
        handle.write("# End-to-end selection benchmark\n\n")
        handle.write(
            "Timed scope: detector inference + feature extraction + acquisition scoring + sample selection. "
            "Model loading, warm-up and result-artifact writes are excluded.\n\n"
        )
        selected_columns = [
            "Method", "Params_M", "GMACs", "GFLOPs", "Pool_images", "Selection_budget",
            "Mean_total_ms", "Std_total_ms", "Mean_per_image_ms", "Std_per_image_ms",
        ]
        handle.write("| " + " | ".join(selected_columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(selected_columns)) + "|\n")
        for row in rows:
            values = [str(row[column]) for column in selected_columns]
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    dataset_cache: dict[str, DatasetPool] = {}

    for config_arg in args.configs:
        config_path = Path(config_arg)
        config = read_yaml(config_path)
        if args.dataset_yaml is not None:
            config["dataset_yaml"] = args.dataset_yaml
        dataset_key = str(config.get("dataset_yaml"))
        if dataset_key not in dataset_cache:
            dataset_cache[dataset_key] = load_dataset_pool(config)
        pool = dataset_cache[dataset_key]
        labeled_size = args.labeled_size if args.labeled_size is not None else int(config["initial_labeled_count"])
        budget = args.selection_budget if args.selection_budget is not None else int(config["samples_per_round"])
        labeled_indices, candidate_indices = shared_split(pool, labeled_size, args.pool_size, args.seed)
        if budget > len(candidate_indices):
            raise ValueError(f"Selection budget {budget} exceeds pool size {len(candidate_indices)}")

        model_name = args.model or config.get("model_name", "yolo11n.pt")
        print(f"Loading {model_name} for {config_path} ...")
        model = create_model(str(model_name), categories=pool.class_names)

        detector_params = detector_gmacs = detector_gflops = 0.0
        extra_params = extra_gmacs = extra_gflops = 0.0
        extra_name = ""
        if not args.skip_profile:
            detector_params, detector_gmacs, detector_gflops = profile_detector(
                model, args.imgsz or int(config.get("imgsz", 640))
            )
            extra_params, extra_gmacs, extra_gflops, extra_name = profile_extra_models(
                config, len(pool.class_names)
            )

        for iteration in range(args.warmup):
            print(f"Warm-up {iteration + 1}/{args.warmup}: {config_path.name}")
            with tempfile.TemporaryDirectory(prefix="fdal_selection_benchmark_") as temporary_dir:
                run_query(
                    config, model, candidate_indices, pool, labeled_indices, budget, args.device,
                    temporary_dir,
                )

        elapsed_ms: list[float] = []
        for iteration in range(args.repeats):
            # Directory construction and removal are outside the timed region.
            # FDAL may create temporary object crops inside query; that work is
            # intentionally included because it is required by acquisition.
            with tempfile.TemporaryDirectory(prefix="fdal_selection_benchmark_") as temporary_dir:
                _synchronise_cuda()
                start = time.perf_counter()
                selected_indices = run_query(
                    config, model, candidate_indices, pool, labeled_indices, budget, args.device,
                    temporary_dir,
                )
                _synchronise_cuda()
                elapsed = (time.perf_counter() - start) * 1000.0
            if len(selected_indices) != budget:
                raise RuntimeError(
                    f"{config_path.name} returned {len(selected_indices)} selections; expected {budget}"
                )
            elapsed_ms.append(elapsed)
            print(f"Timed run {iteration + 1}/{args.repeats}: {elapsed:.2f} ms")

        total_params = int(detector_params + extra_params)
        total_gmacs = detector_gmacs + extra_gmacs
        total_gflops = detector_gflops + extra_gflops
        method = " -> ".join(_strategy_names(config))
        rows.append(
            {
                "Method": method,
                "Config": str(config_path),
                "Model": str(model_name),
                "Extra_model": extra_name or "None",
                "Params_M": f"{total_params / 1e6:.2f}",
                "GMACs": f"{total_gmacs:.2f}",
                "GFLOPs": f"{total_gflops:.2f}",
                "Pool_images": len(candidate_indices),
                "Selection_budget": budget,
                "Warmup": args.warmup,
                "Repeats": args.repeats,
                "Mean_total_ms": f"{sum(elapsed_ms) / len(elapsed_ms):.2f}",
                "Std_total_ms": f"{population_std(elapsed_ms):.2f}",
                "Mean_per_image_ms": f"{sum(elapsed_ms) / len(elapsed_ms) / len(candidate_indices):.4f}",
                "Std_per_image_ms": f"{population_std(elapsed_ms) / len(candidate_indices):.4f}",
            }
        )

    output_dir = Path(args.output_dir)
    write_outputs(rows, output_dir)
    print(f"CSV: {output_dir / 'selection_benchmark.csv'}")
    print(f"Markdown: {output_dir / 'selection_benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
