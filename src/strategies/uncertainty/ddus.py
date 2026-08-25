import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy


class DDUSStrategy(BaseStrategy):
    """Detection Difficulty-aware Uncertainty Sampling."""

    def __init__(
        self,
        model,
        lambda_cls: float = 0.6,
        lambda_loc: float = 0.4,
        rho: float = 0.5,
        top_m: int = 3,
        sampling_conf: float = 0.1,
        synthetic_class_count: Optional[int] = None,
        experiment_dir: Optional[str] = None,
        round: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        if lambda_cls < 0 or lambda_loc < 0:
            raise ValueError("lambda_cls and lambda_loc must be non-negative")
        if not math.isclose(lambda_cls + lambda_loc, 1.0, abs_tol=1e-6):
            raise ValueError("lambda_cls and lambda_loc must sum to 1")
        if rho < 0:
            raise ValueError("rho must be non-negative")
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        if not 0.0 <= sampling_conf <= 1.0:
            raise ValueError("sampling_conf must be in [0, 1]")

        self.lambda_cls = float(lambda_cls)
        self.lambda_loc = float(lambda_loc)
        self.rho = float(rho)
        self.top_m = int(top_m)
        self.sampling_conf = float(sampling_conf)
        self.synthetic_class_count = synthetic_class_count
        self.experiment_dir = experiment_dir
        self.round = round
        self.classwise_cls_quality: Optional[np.ndarray] = None
        self.classwise_loc_quality: Optional[np.ndarray] = None

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        if self.synthetic_class_count is not None:
            if self.synthetic_class_count <= 0:
                raise ValueError("synthetic_class_count must be positive")
            self.classwise_cls_quality = np.full(self.synthetic_class_count, 0.5, dtype=float)
            self.classwise_loc_quality = np.full(self.synthetic_class_count, 0.5, dtype=float)
            logger.info(
                f"Using neutral synthetic DDUS quality for {self.synthetic_class_count} classes"
            )
        else:
            self.classwise_cls_quality, self.classwise_loc_quality = self._load_quality_vectors()

        local_kwargs = dict(kwargs)
        num_inf = int(local_kwargs.pop("num_inference", -1))
        candidate_indices = unlabeled_indices[:num_inf] if num_inf > 0 else unlabeled_indices
        candidate_paths = self._get_image_paths_for_indices(candidate_indices, image_paths)

        print(f"Computing DDUS uncertainty scores for {len(candidate_paths)} images...")
        start_time = time.time()
        results = self.model.inference(
            candidate_paths,
            return_boxes=True,
            return_probs=True,
            return_classes=True,
            num_inference=-1,
            conf=self.sampling_conf,
            inference_device=self.strategy_params.get(
                "inference_device", self.strategy_params.get("device", "auto")
            ),
            inference_batch_size=self.strategy_params.get("inference_batch_size", 1),
            **local_kwargs,
        )

        processed_indices = candidate_indices[: len(results)]
        scores = np.array([self._compute_ddus_score(result) for result in results], dtype=float)
        n_selected = min(n_samples, len(processed_indices))
        top_local = np.argsort(scores, kind="stable")[-n_selected:]
        selected_indices = processed_indices[top_local]

        elapsed = time.time() - start_time
        self._write_time_log(elapsed, len(processed_indices))
        if scores.size:
            selected_scores = np.sort(scores[top_local])
            print(f"Top 5 DDUS uncertainty scores: {selected_scores[-5:]}")
            print(
                f"DDUS score statistics - Mean: {scores.mean():.4f}, "
                f"Std: {scores.std():.4f}, Min: {scores.min():.4f}, "
                f"Max: {scores.max():.4f}"
            )

        return selected_indices

    def _load_quality_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        search_dirs = []
        model_path = getattr(self.model, "model_path", None)
        if model_path:
            search_dirs.append(Path(model_path).parent)
        if self.experiment_dir is not None and self.round is not None:
            train_dir = Path(self.experiment_dir) / f"round_{self.round - 1}" / "train"
            search_dirs.extend(path.parent for path in train_dir.glob("*/weights/last.pt"))
            search_dirs.extend(path.parent for path in train_dir.glob("*/weights/best.pt"))

        checked = []
        for quality_dir in dict.fromkeys(search_dirs):
            cls_paths = [
                quality_dir / "classwise_quality_cls.npy",
                quality_dir / "classwise_cls_quality.npy",
            ]
            loc_paths = [
                quality_dir / "classwise_quality_loc.npy",
                quality_dir / "classwise_loc_quality.npy",
            ]
            checked.extend(str(path) for path in cls_paths + loc_paths)
            cls_path = next((path for path in cls_paths if path.exists()), None)
            loc_path = next((path for path in loc_paths if path.exists()), None)
            if cls_path is not None and loc_path is not None:
                cls_quality = np.asarray(np.load(cls_path), dtype=float).reshape(-1)
                loc_quality = np.asarray(np.load(loc_path), dtype=float).reshape(-1)
                if cls_quality.shape != loc_quality.shape:
                    raise ValueError(
                        "DDUS classification and localization quality vectors have different shapes"
                    )
                logger.info(f"Loaded DDUS quality vectors from {quality_dir}")
                return np.clip(cls_quality, 0.0, 1.0), np.clip(loc_quality, 0.0, 1.0)

        raise FileNotFoundError(
            "DDUS quality vectors were not produced by detector training. "
            "Run training with a strategy containing 'ddus'. Checked: "
            + ", ".join(checked)
        )

    def _get_classwise_weights(self) -> Dict[int, float]:
        if self.classwise_cls_quality is None or self.classwise_loc_quality is None:
            raise ValueError("DDUS quality vectors have not been loaded")

        difficulty = (
            self.lambda_cls * (1.0 - self.classwise_cls_quality)
            + self.lambda_loc * (1.0 - self.classwise_loc_quality)
        )
        raw_weights = 1.0 + self.rho * difficulty
        weights = raw_weights / max(float(raw_weights.mean()), 1e-12)
        return {class_id: float(weight) for class_id, weight in enumerate(weights)}

    def _compute_ddus_score(self, result) -> float:
        if result.probs is None or len(result.probs) == 0:
            return 0.0

        confidences = np.asarray(result.probs, dtype=float).reshape(-1)
        uncertainties = self._binary_entropy(confidences)
        classes = (
            np.asarray(result.classes, dtype=int).reshape(-1)
            if result.classes is not None
            else np.zeros(len(uncertainties), dtype=int)
        )
        if len(classes) != len(uncertainties):
            raise ValueError(
                f"DDUS received {len(classes)} classes for {len(uncertainties)} detections"
            )

        class_weights = self._get_classwise_weights()
        weights = np.array([class_weights.get(int(class_id), 1.0) for class_id in classes])
        object_scores = uncertainties * weights
        top_count = min(self.top_m, len(object_scores))
        top_scores = np.partition(object_scores, len(object_scores) - top_count)[-top_count:]
        return float(top_scores.mean())

    @staticmethod
    def _binary_entropy(confidences: np.ndarray) -> np.ndarray:
        confidences = np.clip(np.asarray(confidences, dtype=float), 1e-8, 1.0 - 1e-8)
        entropy = -(
            confidences * np.log(confidences)
            + (1.0 - confidences) * np.log(1.0 - confidences)
        )
        return entropy / math.log(2.0)

    def _write_time_log(self, elapsed: float, num_images: int) -> None:
        if not self.experiment_dir:
            return
        log_name = os.environ.get("TIME_LOGFILE", "time_log.txt")
        log_path = Path(self.experiment_dir) / log_name
        if not log_path.exists():
            log_path.write_text("Round,TotalTime,NumImages,TimePerImage\n")
        time_per_image = elapsed / num_images if num_images else 0.0
        with log_path.open("a") as log_file:
            log_file.write(
                f"{self.round},{elapsed:.4f},{num_images},{time_per_image:.4f}\n"
            )

    def get_strategy_name(self) -> str:
        return "ddus"
