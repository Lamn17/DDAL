"""Benchmark matched TensorRT FP32, FP16 and calibrated INT8 YOLO engines.

This measures detector validation and prediction latency only.  It deliberately
does not benchmark active-learning selection: strategies such as ``recal`` need
PyTorch feature hooks and pre-NMS class probabilities, which are not exposed by
an exported TensorRT engine in the current model wrapper.  All precision rows
use the same TensorRT backend so differences are not confounded by comparing
native PyTorch inference against TensorRT inference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import re
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METRICS = (
    "model_size_mb",
    "map50",
    "map50_95",
    "metric_precision",
    "recall",
    "delta_map50_95",
    "mean_latency_ms_per_image",
    "std_latency_ms_per_image",
    "p50_latency_ms_per_image",
    "p95_latency_ms_per_image",
    "fps",
    "peak_vram_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the detector-level FP32/FP16/INT8 accuracy-latency trade-off."
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One .pt checkpoint, or one checkpoint per seed when --seeds is supplied.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        help="Optional labels for multi-seed checkpoints, for example: 1 10 100.",
    )
    parser.add_argument(
        "--data", required=True, help="YOLO dataset YAML used for validation and INT8 calibration."
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("fp32", "fp16", "int8"),
        default=("fp32", "fp16", "int8"),
        help="Precisions to test (default: fp32 fp16 int8).",
    )
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Prediction and validation batch size.")
    parser.add_argument("--images", type=int, default=200, help="Validation images timed per repeat.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of the dataset used by Ultralytics during INT8 calibration.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=2.0,
        help="Maximum TensorRT builder workspace in GiB (default: 2).",
    )
    for precision in ("fp32", "fp16", "int8"):
        parser.add_argument(
            f"--{precision}-engine",
            help=f"Reuse an existing TensorRT {precision.upper()} engine instead of exporting it.",
        )
    parser.add_argument(
        "--rebuild-engines",
        action="store_true",
        help="Re-export engines already present under OUTPUT_DIR/engines.",
    )
    parser.add_argument("--skip-val", action="store_true", help="Skip mAP validation.")
    parser.add_argument(
        "--output-dir",
        default="outputs/quantization_benchmark",
        help="Report directory; in multi-seed mode, this is the parent directory.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="In multi-seed mode, aggregate existing per-seed JSON reports without benchmarking.",
    )
    args = parser.parse_args()
    if args.images <= 0 or args.warmup < 0 or args.repeats <= 0 or args.batch <= 0:
        parser.error("images, batch and repeats must be positive; warmup cannot be negative")
    if not 0.0 < args.calibration_fraction <= 1.0:
        parser.error("calibration-fraction must be in (0, 1]")
    if args.workspace <= 0:
        parser.error("workspace must be positive")
    args.precisions = list(dict.fromkeys(args.precisions))
    if args.seeds is not None and len(args.seeds) != len(args.weights):
        parser.error("--seeds must contain exactly one label per checkpoint")
    if args.aggregate_only and len(args.weights) < 2:
        parser.error("--aggregate-only requires multiple checkpoints")
    return args


def read_dataset_yaml(data_path: str) -> dict[str, Any]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    return value


def _paths_from_value(value: Any, root: Path) -> Iterable[Path]:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, str):
            continue
        path = Path(item)
        yield path if path.is_absolute() else root / path


def validation_images(data_path: str, limit: int) -> list[str]:
    """Resolve up to ``limit`` validation images from a standard YOLO YAML."""
    config = read_dataset_yaml(data_path)
    yaml_parent = Path(data_path).resolve().parent
    root_value = config.get("path", yaml_parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = yaml_parent / root
    resolved: list[Path] = []
    for source in _paths_from_value(config.get("val"), root):
        if source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                item = Path(line.strip())
                if item:
                    resolved.append(item if item.is_absolute() else root / item)
        elif source.is_dir():
            resolved.extend(
                sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(source)
    images = [str(path) for path in resolved if path.exists()]
    if not images:
        raise FileNotFoundError("Could not resolve validation images from the dataset YAML 'val' field.")
    return images[:limit]


def synchronise(device: str) -> None:
    if str(device).lower() != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaMemoryMonitor:
    """Sample incremental device memory, including allocations outside PyTorch."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak = 0

    @staticmethod
    def _used_bytes() -> int:
        free, total = torch.cuda.mem_get_info()
        return int(total - free)

    def start(self) -> None:
        self._baseline = self._used_bytes()
        self._peak = self._baseline

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._peak = max(self._peak, self._used_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._peak = max(self._peak, self._used_bytes())
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return max(0.0, (self._peak - self._baseline) / (1024 * 1024))


def require_tensorrt(device: str) -> None:
    """Fail early instead of silently falling back to another backend."""
    if str(device).lower() == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT benchmarking requires a CUDA device.")
    if importlib.util.find_spec("tensorrt") is None:
        raise RuntimeError(
            "TensorRT is not installed. Install a TensorRT version compatible with the CUDA/PyTorch "
            "environment, then re-run the benchmark."
        )


def engine_for_precision(
    args: argparse.Namespace, precision: str, output_dir: Path
) -> Path:
    """Return a supplied/cached engine or safely export one in the report tree."""
    supplied = getattr(args, f"{precision}_engine")
    if supplied:
        path = Path(supplied).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Supplied {precision.upper()} engine not found: {path}")
        return path

    engine_dir = output_dir / "engines"
    engine_dir.mkdir(parents=True, exist_ok=True)
    destination = engine_dir / f"{Path(args.weight).stem}_{precision}.engine"
    if destination.exists() and not args.rebuild_engines:
        return destination

    # Export from a staged checkpoint because Ultralytics always writes
    # <weights-stem>.engine next to the checkpoint. This avoids overwriting a
    # user's existing best.engine beside the source weights.
    build_dir = output_dir / "build" / precision
    build_dir.mkdir(parents=True, exist_ok=True)
    staged_weights = build_dir / Path(args.weight).name
    shutil.copy2(args.weight, staged_weights)

    export_args: dict[str, Any] = {
        "format": "engine",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workspace": args.workspace,
        "half": precision == "fp16",
        "int8": precision == "int8",
    }
    if precision == "int8":
        export_args.update(
            data=str(Path(args.data).resolve()),
            fraction=args.calibration_fraction,
        )
    exported = Path(YOLO(str(staged_weights)).export(**export_args)).resolve()
    if not exported.exists():
        raise RuntimeError(f"Ultralytics did not create the {precision.upper()} engine")
    destination.unlink(missing_ok=True)
    shutil.move(str(exported), destination)
    return destination.resolve()


def validate(model_path: str, args: argparse.Namespace) -> dict[str, float]:
    metrics = YOLO(model_path, task="detect").val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "metric_precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def latency_metrics(
    model_path: str, images: list[str], args: argparse.Namespace
) -> dict[str, float]:
    """Measure synchronized end-to-end latency in fixed-size engine batches."""
    usable_count = len(images) - (len(images) % args.batch)
    if usable_count <= 0:
        raise ValueError(
            f"Latency benchmark needs at least {args.batch} images for batch={args.batch}"
        )
    timed_images = images[:usable_count]
    batches = [
        timed_images[start : start + args.batch]
        for start in range(0, usable_count, args.batch)
    ]

    gc.collect()
    torch.cuda.empty_cache()
    monitor = CudaMemoryMonitor()
    monitor.start()
    measurements: list[float] = []
    total_seconds = 0.0
    try:
        model = YOLO(model_path, task="detect")
        warmup_batch = batches[0]
        for _ in range(args.warmup):
            model.predict(
                warmup_batch,
                imgsz=args.imgsz,
                device=args.device,
                batch=args.batch,
                verbose=False,
            )
        for _ in range(args.repeats):
            for batch_paths in batches:
                synchronise(args.device)
                start = time.perf_counter()
                model.predict(
                    batch_paths,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=args.batch,
                    verbose=False,
                )
                synchronise(args.device)
                elapsed = time.perf_counter() - start
                total_seconds += elapsed
                measurements.append(elapsed * 1000.0 / len(batch_paths))
    finally:
        peak_vram_mb = monitor.stop()

    return {
        "mean_latency_ms_per_image": float(statistics.mean(measurements)),
        "std_latency_ms_per_image": float(
            statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        ),
        "p50_latency_ms_per_image": float(np.percentile(measurements, 50)),
        "p95_latency_ms_per_image": float(np.percentile(measurements, 95)),
        "fps": float((usable_count * args.repeats) / total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
    }


def add_map_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next(
        (
            row.get("map50_95")
            for row in rows
            if row["precision"] == "fp32" and row["status"] == "ok"
        ),
        None,
    )
    for row in rows:
        value = row.get("map50_95")
        row["delta_map50_95"] = (
            round(float(value) - float(baseline), 6)
            if value is not None and baseline is not None
            else None
        )


def _format_metric(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# TensorRT quantization benchmark",
        "",
        "All rows use TensorRT with the same detector checkpoint and evaluation settings.",
        "Latency is end-to-end `predict()` wall time per image, including preprocessing and NMS.",
        "",
        "## Accuracy",
        "",
        "| Precision | mAP@50 | mAP@50:95 | Precision | Recall | ΔmAP@50:95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = "—" if row["precision"] == "fp32" else _format_metric(
            row.get("delta_map50_95"), signed=True
        )
        lines.append(
            "| {precision} | {map50} | {map50_95} | {metric_precision} | {recall} | "
            "{delta} |".format(
                precision=row["precision"].upper(),
                map50=_format_metric(row.get("map50")),
                map50_95=_format_metric(row.get("map50_95")),
                metric_precision=_format_metric(row.get("metric_precision")),
                recall=_format_metric(row.get("recall")),
                delta=delta,
            )
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "Peak VRAM is incremental device memory above the pre-load baseline and includes "
            "TensorRT allocations outside the PyTorch allocator.",
            "",
            "| Precision | Latency mean ± std (ms/image) | P50 | P95 | FPS | Peak VRAM (MB) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        mean = row.get("mean_latency_ms_per_image")
        deviation = row.get("std_latency_ms_per_image")
        latency = (
            f"{float(mean):.3f} ± {float(deviation):.3f}"
            if mean is not None and deviation is not None
            else "—"
        )
        lines.append(
            "| {precision} | {latency} | {p50} | {p95} | {fps} | {vram} |".format(
                precision=row["precision"].upper(),
                latency=latency,
                p50=_format_metric(row.get("p50_latency_ms_per_image")),
                p95=_format_metric(row.get("p95_latency_ms_per_image")),
                fps=_format_metric(row.get("fps")),
                vram=_format_metric(row.get("peak_vram_mb")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "precision",
        "backend",
        "status",
        "model",
        "model_size_mb",
        "map50",
        "map50_95",
        "metric_precision",
        "recall",
        "delta_map50_95",
        "mean_latency_ms_per_image",
        "std_latency_ms_per_image",
        "p50_latency_ms_per_image",
        "p95_latency_ms_per_image",
        "fps",
        "peak_vram_mb",
        "notes",
    ]
    with (output_dir / "quantization_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "quantization_benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (output_dir / "quantization_benchmark.md").write_text(
        markdown_report(rows), encoding="utf-8"
    )


def run_single_benchmark(args: argparse.Namespace, weight: Path, output_dir: Path) -> None:
    """Benchmark one checkpoint and write the existing per-seed report format."""
    if not weight.exists():
        raise FileNotFoundError(f"Weights not found: {weight}")
    args.weight = str(weight)
    require_tensorrt(args.device)
    images = validation_images(args.data, args.images)
    rows: list[dict[str, Any]] = []
    for precision in args.precisions:
        row: dict[str, Any] = {
            "precision": precision,
            "backend": "TensorRT",
            "status": "ok",
            "notes": "",
        }
        try:
            path = engine_for_precision(args, precision, output_dir)
            row["model"] = str(path)
            row["model_size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)
            if args.skip_val:
                for field in ("map50", "map50_95", "metric_precision", "recall"):
                    row[field] = None
            else:
                row.update(validate(str(path), args))
            latency = latency_metrics(str(path), images, args)
            row.update({name: round(value, 4) for name, value in latency.items()})
        except Exception as error:  # Preserve completed rows if another precision fails.
            row.update({"status": "unavailable", "notes": str(error)})
            for field in (
                "model",
                "model_size_mb",
                "map50",
                "map50_95",
                "metric_precision",
                "recall",
                "mean_latency_ms_per_image",
                "std_latency_ms_per_image",
                "p50_latency_ms_per_image",
                "p95_latency_ms_per_image",
                "fps",
                "peak_vram_mb",
            ):
                row.setdefault(field, None)
        rows.append(row)
        message = f"{precision}: {row['status']}"
        if row["notes"]:
            message += f" ({row['notes']})"
        print(message)
    add_map_deltas(rows)
    write_reports(rows, output_dir)
    print(f"Saved quantization benchmark to {output_dir.resolve()}")


def seed_labels(weights: list[str], seeds: list[str] | None) -> list[str]:
    raw_labels = seeds or [Path(weight).stem for weight in weights]
    labels = [re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") for value in raw_labels]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Seed labels must be non-empty and unique")
    return labels


def load_seed_rows(report_path: Path, label: str, precisions: list[str]) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report for seed {label}: {report_path}")
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Invalid report format in {report_path}")
    rows = {str(row.get("precision")): row for row in value if isinstance(row, dict)}
    missing = set(precisions).difference(rows)
    if missing:
        raise ValueError(f"Seed {label} report is missing precisions: {sorted(missing)}")
    unavailable = [precision for precision in precisions if rows[precision].get("status") != "ok"]
    if unavailable:
        raise RuntimeError(f"Seed {label} has unavailable precisions: {', '.join(unavailable)}")
    return rows


def aggregate_seed_rows(
    seed_rows: dict[str, dict[str, dict[str, Any]]], precisions: list[str]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for precision in precisions:
        rows = [seed_rows[label][precision] for label in seed_rows]
        aggregate: dict[str, Any] = {
            "precision": precision,
            "seeds": list(seed_rows),
            "n_seeds": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            aggregate[f"{metric}_n"] = len(values)
            aggregate[f"{metric}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        results.append(aggregate)
    return results


def format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row[f"{metric}_mean"]
    deviation = row[f"{metric}_std"]
    count = row[f"{metric}_n"]
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {deviation:.4f}" if deviation is not None else f"{mean:.4f} (n={count})"


def write_multiseed_reports(
    rows: list[dict[str, Any]], seed_rows: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantization_multiseed.json").write_text(
        json.dumps({"per_seed": seed_rows, "aggregate": rows}, indent=2), encoding="utf-8"
    )
    with (output_dir / "quantization_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# TensorRT quantization benchmark across seeds",
        "",
        "Values are mean ± sample standard deviation across checkpoints; no best seed is selected.",
        "",
        "| Precision | mAP50:95 | Latency (ms/image) | P50 (ms) | P95 (ms) | FPS | VRAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {precision} | {map} | {latency} | {p50} | {p95} | {fps} | {vram} |".format(
                precision=str(row["precision"]).upper(),
                map=format_mean_std(row, "map50_95"),
                latency=format_mean_std(row, "mean_latency_ms_per_image"),
                p50=format_mean_std(row, "p50_latency_ms_per_image"),
                p95=format_mean_std(row, "p95_latency_ms_per_image"),
                fps=format_mean_std(row, "fps"),
                vram=format_mean_std(row, "peak_vram_mb"),
            )
        )
    (output_dir / "quantization_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.weights) == 1:
        if args.seeds is not None:
            raise ValueError("--seeds is only needed when benchmarking multiple checkpoints")
        if args.aggregate_only:
            raise ValueError("--aggregate-only requires multiple checkpoints")
        run_single_benchmark(args, Path(args.weights[0]), Path(args.output_dir).resolve())
        return

    labels = seed_labels(args.weights, args.seeds)
    output_dir = Path(args.output_dir).resolve()
    seed_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for label, weight in zip(labels, args.weights, strict=True):
        checkpoint = Path(weight)
        seed_dir = output_dir / f"seed_{label}"
        if not args.aggregate_only:
            print(f"\n=== Benchmarking seed {label}: {checkpoint} ===", flush=True)
            run_single_benchmark(args, checkpoint, seed_dir)
        seed_rows[label] = load_seed_rows(
            seed_dir / "quantization_benchmark.json", label, args.precisions
        )
    aggregate_rows = aggregate_seed_rows(seed_rows, args.precisions)
    write_multiseed_reports(aggregate_rows, seed_rows, output_dir)
    print(f"\nSaved aggregate report to {output_dir / 'quantization_multiseed.md'}")


if __name__ == "__main__":
    main()
