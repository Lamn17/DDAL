from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from ..base import BaseStrategy


def normalized_posterior_entropy(
    probabilities: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Compute normalized Shannon entropy for categorical posteriors."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (num_boxes, num_classes)")
    if values.shape[1] <= 1:
        return np.zeros(len(values), dtype=float)
    values = np.clip(values, epsilon, None)
    values /= values.sum(axis=1, keepdims=True)
    return -np.sum(values * np.log(values + epsilon), axis=1) / np.log(values.shape[1])


def normalized_dfl_entropy(
    distributions: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Compute mean four-side normalized entropy from DFL distributions."""
    values = np.asarray(distributions, dtype=float)
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("DFL distributions must have shape (num_boxes, 4, reg_max)")
    reg_max = values.shape[2]
    if reg_max <= 1:
        return np.zeros(len(values), dtype=float)
    values = np.clip(values, epsilon, None)
    values /= values.sum(axis=2, keepdims=True)
    side_entropy = -np.sum(values * np.log(values + epsilon), axis=2) / np.log(reg_max)
    return side_entropy.mean(axis=1)


def compute_aspect_difficulty(
    class_quality: np.ndarray, localization_quality: np.ndarray
) -> np.ndarray:
    """Run Difficulty Class Quality Estimation (DCQE)."""
    cls = np.asarray(class_quality, dtype=float).reshape(-1)
    loc = np.asarray(localization_quality, dtype=float).reshape(-1)
    if cls.shape != loc.shape:
        raise ValueError("classification and localization quality shapes must match")
    return np.stack(
        (1.0 - np.clip(cls, 0.0, 1.0), 1.0 - np.clip(loc, 0.0, 1.0)), axis=1
    )


def compute_aspect_yield(
    class_probabilities: np.ndarray,
    localization_uncertainty: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Run Dual Entropy Class Scoring (DECS) for one image."""
    probabilities = np.asarray(class_probabilities, dtype=float)
    localization = np.asarray(localization_uncertainty, dtype=float).reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError("class probabilities must be a 2-D matrix")
    if len(probabilities) != len(localization):
        raise ValueError("class probabilities and localization uncertainty must align")
    if not len(probabilities):
        return np.zeros((probabilities.shape[1], 2), dtype=float)
    probabilities = np.clip(probabilities, epsilon, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classification = normalized_posterior_entropy(probabilities, epsilon)
    localization = np.clip(localization, 0.0, 1.0)
    return np.stack(
        (
            np.sum(probabilities * classification[:, None], axis=0),
            np.sum(probabilities * localization[:, None], axis=0),
        ),
        axis=1,
    )


def compute_aspect_quotas(
    difficulty: np.ndarray,
    budget: int,
    average_labeled_objects: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Allocate the expected instance budget across class-aspect pairs."""
    values = np.asarray(difficulty, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("difficulty must have shape (num_classes, 2)")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not np.isfinite(average_labeled_objects) or average_labeled_objects < 0:
        raise ValueError("average_labeled_objects must be finite and non-negative")
    values = np.clip(values, 0.0, None)
    total_instance_budget = float(budget) * float(average_labeled_objects)
    denominator = float(values.sum())
    if denominator <= epsilon:
        return np.full_like(values, total_instance_budget / max(values.size, 1))
    return total_instance_budget * values / denominator


def saturation_greedy(
    yields: np.ndarray, quotas: np.ndarray, budget: int
) -> tuple[List[int], np.ndarray]:
    """Run Budgeted Class Query Selection (BCQS) with saturation greedy."""
    values = np.asarray(yields, dtype=float)
    limits = np.asarray(quotas, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("yields must have shape (num_images, num_classes, 2)")
    if limits.shape != values.shape[1:]:
        raise ValueError("quotas must have shape (num_classes, 2)")
    if budget <= 0 or budget > len(values):
        raise ValueError("budget must be in [1, num_images]")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("yields must be finite and non-negative")
    if np.any(~np.isfinite(limits)) or np.any(limits < 0):
        raise ValueError("quotas must be finite and non-negative")

    selected: List[int] = []
    selected_mask = np.zeros(len(values), dtype=bool)
    achieved = np.zeros_like(limits)
    for _ in range(budget):
        remaining = np.maximum(limits - achieved, 0.0)
        gains = np.minimum(values, remaining[None, :, :]).sum(axis=(1, 2))
        gains[selected_mask] = -np.inf
        candidate = int(np.argmax(gains))
        selected.append(candidate)
        selected_mask[candidate] = True
        achieved += values[candidate]
    return selected, achieved


class DDALStrategy(BaseStrategy):
    """DCQE -> DECS -> BCQS acquisition for active object detection."""

    def __init__(
        self,
        model,
        experiment_dir: Optional[str] = None,
        round: Optional[int] = None,
        labeled_indices: Optional[Sequence[int]] = None,
        sampling_conf: float = 0.25,
        quality_initial: float = 0.5,
        difficulty_mode: str = "dual",
        signal_mode: str = "both",
        selection_mode: str = "bcqs",
        label_paths: Optional[Sequence[Optional[str]]] = None,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        if not 0.0 <= sampling_conf <= 1.0:
            raise ValueError("sampling_conf must be in [0, 1]")
        if not 0.0 <= quality_initial <= 1.0:
            raise ValueError("quality_initial must be in [0, 1]")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if difficulty_mode not in {"dual", "uniform", "combined"}:
            raise ValueError(
                "difficulty_mode must be one of: dual, uniform, combined"
            )
        if signal_mode not in {"both", "cls", "loc"}:
            raise ValueError("signal_mode must be one of: both, cls, loc")
        if selection_mode not in {"bcqs", "topk"}:
            raise ValueError("selection_mode must be one of: bcqs, topk")
        self.experiment_dir = experiment_dir
        self.round = round
        self.labeled_indices = np.asarray(
            labeled_indices if labeled_indices is not None else [], dtype=int
        )
        self.sampling_conf = float(sampling_conf)
        self.quality_initial = float(quality_initial)
        self.difficulty_mode = difficulty_mode
        self.signal_mode = signal_mode
        self.selection_mode = selection_mode
        self.label_paths = list(label_paths) if label_paths is not None else None
        self.epsilon = float(epsilon)
        self._warned_dfl_fallback = False
        self._warned_quality_fallback = False

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        self._all_image_paths = image_paths
        local_kwargs = dict(kwargs)
        num_inference = int(local_kwargs.pop("num_inference", -1))
        candidate_indices = (
            unlabeled_indices[:num_inference]
            if num_inference > 0
            else unlabeled_indices
        )
        if len(candidate_indices) < n_samples:
            raise ValueError(
                f"DDAL received {len(candidate_indices)} inferred images for a budget "
                f"of {n_samples}. Increase num_inference or reduce n_samples."
            )

        start_time = time.time()
        candidate_paths = self._get_image_paths_for_indices(
            candidate_indices, image_paths
        )
        results = self._inference(candidate_paths, local_kwargs)
        processed_count = min(len(results), len(candidate_indices))
        if processed_count < n_samples:
            raise RuntimeError(
                f"DDAL only received {processed_count} inference results for budget {n_samples}"
            )
        processed_indices = candidate_indices[:processed_count]
        results = results[:processed_count]

        num_classes = self._num_classes(results)
        cls_quality, loc_quality = self._load_aspect_quality(num_classes)
        difficulty = self._ablation_difficulty(cls_quality, loc_quality)
        yields = np.asarray(
            [self._image_aspect_yield(result, num_classes) for result in results],
            dtype=float,
        )
        difficulty, yields = self._apply_signal_mode(difficulty, yields)
        labeled_indices = self._usable_labeled_indices(image_paths, processed_indices)
        average_objects = self._average_labeled_objects(labeled_indices, num_classes)
        quotas = compute_aspect_quotas(
            difficulty, n_samples, average_objects, self.epsilon
        )

        selection_start = time.perf_counter()
        selected_local_list, achieved = self._select(yields, quotas, n_samples)
        selection_elapsed = time.perf_counter() - selection_start
        selected_local = np.asarray(selected_local_list, dtype=int)
        selected_indices = processed_indices[selected_local]

        self._write_artifacts(
            time.time() - start_time,
            processed_count,
            selected_indices,
            image_paths,
            results,
            processed_indices,
        )
        self._write_ddal_state(
            cls_quality, loc_quality, difficulty, quotas, achieved, average_objects
        )
        self._write_allocation_log(
            selected_indices, difficulty, quotas, achieved, num_classes
        )
        self._write_selection_metrics(
            selection_elapsed,
            processed_count,
            average_objects,
            quotas,
            achieved,
            results,
        )
        logger.info(
            "DDAL selected {} images from {} candidates "
            "(difficulty={}, signal={}, selection={}, mean labeled objects {:.4f}, "
            "quota coverage {:.4f}).",
            len(selected_indices),
            processed_count,
            self.difficulty_mode,
            self.signal_mode,
            self.selection_mode,
            average_objects,
            self._quota_coverage(achieved, quotas),
        )
        return selected_indices

    def _ablation_difficulty(
        self, cls_quality: np.ndarray, loc_quality: np.ndarray
    ) -> np.ndarray:
        """Build class-aspect difficulty for full and DCQE ablations."""
        if self.difficulty_mode == "uniform":
            return np.ones((len(cls_quality), 2), dtype=float)
        if self.difficulty_mode == "combined":
            combined = 1.0 - np.clip(cls_quality, 0.0, 1.0) * np.clip(
                loc_quality, 0.0, 1.0
            )
            return np.repeat(combined[:, None], 2, axis=1)
        return compute_aspect_difficulty(cls_quality, loc_quality)

    def _apply_signal_mode(
        self, difficulty: np.ndarray, yields: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep both DECS channels or isolate classification/localization."""
        difficulty = np.asarray(difficulty, dtype=float).copy()
        yields = np.asarray(yields, dtype=float).copy()
        if self.signal_mode == "cls":
            difficulty[:, 1] = 0.0
            yields[:, :, 1] = 0.0
        elif self.signal_mode == "loc":
            difficulty[:, 0] = 0.0
            yields[:, :, 0] = 0.0
        return difficulty, yields

    def _select(
        self, yields: np.ndarray, quotas: np.ndarray, budget: int
    ) -> tuple[List[int], np.ndarray]:
        """Run full BCQS or its unsaturated top-k ablation."""
        if self.selection_mode == "bcqs":
            return saturation_greedy(yields, quotas, budget)
        scores = yields.sum(axis=(1, 2))
        selected = np.argsort(-scores, kind="stable")[:budget].tolist()
        achieved = yields[selected].sum(axis=0)
        return selected, achieved

    def _inference(self, paths: Sequence[str], local_kwargs: dict) -> List:
        if not paths:
            return []
        return self.model.inference(
            list(paths),
            return_boxes=True,
            return_classes=True,
            return_probs=True,
            return_logits=True,
            return_class_probs=True,
            return_localization_uncertainty=True,
            return_features=False,
            num_inference=-1,
            conf=self.sampling_conf,
            inference_device=self.strategy_params.get(
                "inference_device", self.strategy_params.get("device", "auto")
            ),
            inference_batch_size=self.strategy_params.get("inference_batch_size", 1),
            **local_kwargs,
        )

    def _image_aspect_yield(self, result, num_classes: int) -> np.ndarray:
        probabilities = self._class_probabilities(result, num_classes)
        localization = self._localization_uncertainty(result, len(probabilities))
        return compute_aspect_yield(probabilities, localization, self.epsilon)

    def _class_probabilities(self, result, num_classes: int) -> np.ndarray:
        explicit = getattr(result, "class_probs", None)
        if explicit is not None:
            probabilities = np.asarray(explicit, dtype=float)
        elif getattr(result, "logits", None) is not None:
            logits = np.asarray(result.logits, dtype=float)
            if logits.ndim != 2:
                raise ValueError(
                    "DDAL logits must have shape (num_boxes, num_classes)"
                )
            logits = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
        else:
            raise RuntimeError(
                "DECS requires per-box class probabilities. The detector backend must "
                "provide InferenceResult.class_probs or logits."
            )
        if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
            raise ValueError(
                "DDAL class probabilities must have shape (num_boxes, num_classes)"
            )
        if not len(probabilities):
            return np.empty((0, num_classes), dtype=float)
        probabilities = np.clip(probabilities, self.epsilon, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _localization_uncertainty(self, result, num_boxes: int) -> np.ndarray:
        dfl_distributions = getattr(result, "dfl_probs", None)
        if dfl_distributions is not None:
            values = normalized_dfl_entropy(dfl_distributions, self.epsilon)
        else:
            explicit = getattr(result, "localization_uncertainty", None)
            values = (
                np.asarray(explicit, dtype=float).reshape(-1)
                if explicit is not None
                else None
            )
        if values is not None:
            if len(values) != num_boxes:
                raise ValueError("DFL uncertainty must align with class probabilities")
            return np.clip(values, 0.0, 1.0)
        if not self._warned_dfl_fallback:
            logger.warning(
                "DDAL could not access DFL distributions; using 1 - detection confidence "
                "as localization uncertainty."
            )
            self._warned_dfl_fallback = True
        confidence_values = getattr(result, "probs", None)
        confidences = np.asarray(
            confidence_values if confidence_values is not None else [], dtype=float
        ).reshape(-1)
        if len(confidences) != num_boxes:
            raise ValueError("Fallback confidence must align with class probabilities")
        return 1.0 - np.clip(confidences, 0.0, 1.0)

    def _load_aspect_quality(self, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
        model_core = getattr(self.model, "model", None)
        cls_value = self._first_attribute(
            model_core, ("classwise_quality_cls", "classwise_cls_quality")
        )
        loc_value = self._first_attribute(
            model_core, ("classwise_quality_loc", "classwise_loc_quality")
        )
        if cls_value is not None and loc_value is not None:
            return (
                self._fit_quality_vector(cls_value, num_classes),
                self._fit_quality_vector(loc_value, num_classes),
            )
        for quality_dir in self._quality_search_dirs():
            for cls_name, loc_name in (
                ("classwise_quality_cls.npy", "classwise_quality_loc.npy"),
                ("classwise_cls_quality.npy", "classwise_loc_quality.npy"),
            ):
                cls_path, loc_path = quality_dir / cls_name, quality_dir / loc_name
                if cls_path.exists() and loc_path.exists():
                    logger.info("DDAL loaded dual-aspect quality from {}", quality_dir)
                    return (
                        self._fit_quality_vector(np.load(cls_path), num_classes),
                        self._fit_quality_vector(np.load(loc_path), num_classes),
                    )
        combined = self._first_attribute(model_core, ("classwise_quality",))
        if combined is None:
            for quality_dir in self._quality_search_dirs():
                path = quality_dir / "classwise_quality.npy"
                if path.exists():
                    combined = np.load(path)
                    break
        if combined is not None:
            if not self._warned_quality_fallback:
                logger.warning(
                    "DDAL dual-aspect quality is missing; duplicating legacy "
                    "classwise_quality for both aspects."
                )
                self._warned_quality_fallback = True
            fitted = self._fit_quality_vector(combined, num_classes)
            return fitted.copy(), fitted.copy()
        initial = np.full(num_classes, self.quality_initial, dtype=float)
        return initial.copy(), initial.copy()

    @staticmethod
    def _first_attribute(obj, names: Sequence[str]):
        if obj is None:
            return None
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return value
        return None

    def _quality_search_dirs(self) -> List[Path]:
        directories: List[Path] = []
        model_path = getattr(self.model, "model_path", None)
        if model_path:
            directories.append(Path(model_path).parent)
        if self.experiment_dir and self.round is not None and self.round > 0:
            train_dir = Path(self.experiment_dir) / f"round_{self.round - 1}" / "train"
            directories.extend(
                path.parent for path in train_dir.glob("*/weights/best.pt")
            )
            directories.extend(
                path.parent for path in train_dir.glob("*/weights/last.pt")
            )
        return list(dict.fromkeys(directories))

    def _fit_quality_vector(self, value, num_classes: int) -> np.ndarray:
        quality = np.asarray(value, dtype=float).reshape(-1)
        fitted = np.full(num_classes, self.quality_initial, dtype=float)
        fitted[: min(num_classes, len(quality))] = quality[:num_classes]
        return np.clip(fitted, 0.0, 1.0)

    def _num_classes(self, results: Iterable) -> int:
        model_names = getattr(getattr(self.model, "model", None), "names", None)
        if isinstance(model_names, (dict, list, tuple)):
            return max(1, len(model_names))
        for result in results:
            probabilities = getattr(result, "class_probs", None)
            if probabilities is not None and np.asarray(probabilities).ndim == 2:
                return max(1, np.asarray(probabilities).shape[1])
        raise RuntimeError("DDAL could not determine the detector class count")

    def _average_labeled_objects(
        self, labeled_indices: Sequence[int], num_classes: int
    ) -> float:
        indices = np.asarray(labeled_indices, dtype=int)
        if not len(indices):
            logger.warning("No labeled indices for mean object count; using 1.0.")
            return 1.0
        return float(
            self._class_instance_counts(indices, num_classes).sum() / len(indices)
        )

    def _class_instance_counts(
        self, indices: Sequence[int], num_classes: int
    ) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=float)
        for index in np.asarray(indices, dtype=int):
            label_path = self._label_path_for_index(int(index))
            if label_path is None or not label_path.exists():
                continue
            try:
                for line in label_path.read_text().splitlines():
                    fields = line.split()
                    if fields:
                        class_id = int(float(fields[0]))
                        if 0 <= class_id < num_classes:
                            counts[class_id] += 1.0
            except (OSError, ValueError) as error:
                logger.warning(
                    "DDAL could not parse label file {}: {}", label_path, error
                )
        return counts

    def _label_path_for_index(self, index: int) -> Optional[Path]:
        if self.label_paths is not None and 0 <= index < len(self.label_paths):
            value = self.label_paths[index]
            return Path(value) if value else None
        image_paths = getattr(self, "_all_image_paths", None)
        if image_paths is None or not 0 <= index < len(image_paths):
            return None
        image_path = Path(image_paths[index])
        parts = list(image_path.parts)
        try:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        except ValueError:
            return None
        return Path(*parts).with_suffix(".txt")

    def _usable_labeled_indices(self, image_paths, candidate_indices) -> np.ndarray:
        labeled = self.labeled_indices
        if not len(labeled) and self.experiment_dir and self.round and self.round > 0:
            metadata_path = (
                Path(self.experiment_dir) / f"round_{self.round - 1}" / "metadata.yaml"
            )
            if metadata_path.exists():
                import yaml

                metadata = yaml.safe_load(metadata_path.read_text()) or {}
                labeled = np.asarray(metadata.get("train_indices", []), dtype=int)
        candidate_set = set(np.asarray(candidate_indices, dtype=int).tolist())
        return np.asarray(
            sorted(
                {
                    int(index)
                    for index in labeled
                    if 0 <= int(index) < len(image_paths)
                    and int(index) not in candidate_set
                }
            ),
            dtype=int,
        )

    @staticmethod
    def _quota_coverage(achieved: np.ndarray, quotas: np.ndarray) -> float:
        total_quota = float(np.asarray(quotas, dtype=float).sum())
        return (
            1.0
            if total_quota <= 0
            else float(np.minimum(achieved, quotas).sum() / total_quota)
        )

    def _write_ddal_state(
        self, cls_quality, loc_quality, difficulty, quotas, achieved, average_objects
    ) -> None:
        if not self.experiment_dir:
            return
        experiment_dir = Path(self.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            experiment_dir / "ddal_state.npz",
            round=np.asarray(int(self.round or 0)),
            difficulty_mode=np.asarray(self.difficulty_mode),
            signal_mode=np.asarray(self.signal_mode),
            selection_mode=np.asarray(self.selection_mode),
            quality_cls=cls_quality,
            quality_loc=loc_quality,
            difficulty_cls=difficulty[:, 0],
            difficulty_loc=difficulty[:, 1],
            quotas=quotas,
            predicted_yield=achieved,
            average_labeled_objects=np.asarray(average_objects),
        )
        path = experiment_dir / "ddal_aspect_dynamics.csv"
        if not path.exists():
            path.write_text("round,class,Q_cls,Q_loc,D_cls,D_loc\n")
        with path.open("a") as handle:
            for class_id in range(len(cls_quality)):
                handle.write(
                    f"{int(self.round or 0)},{class_id},{cls_quality[class_id]:.8g},"
                    f"{loc_quality[class_id]:.8g},{difficulty[class_id, 0]:.8g},"
                    f"{difficulty[class_id, 1]:.8g}\n"
                )

    def _write_allocation_log(
        self, selected_indices, difficulty, quotas, achieved, num_classes
    ) -> None:
        if not self.experiment_dir:
            return
        actual_counts = self._class_instance_counts(selected_indices, num_classes)
        path = Path(self.experiment_dir) / "ddal_allocation_fidelity.csv"
        if not path.exists():
            path.write_text(
                "round,class,aspect,difficulty,quota,predicted_yield,actual_instances\n"
            )
        with path.open("a") as handle:
            for class_id in range(num_classes):
                for aspect_id, aspect in enumerate(("cls", "loc")):
                    handle.write(
                        f"{int(self.round or 0)},{class_id},{aspect},"
                        f"{difficulty[class_id, aspect_id]:.8g},"
                        f"{quotas[class_id, aspect_id]:.8g},"
                        f"{achieved[class_id, aspect_id]:.8g},"
                        f"{actual_counts[class_id]:.0f}\n"
                    )

    def _write_selection_metrics(
        self, elapsed, processed_count, average_objects, quotas, achieved, results
    ) -> None:
        if not self.experiment_dir:
            return
        dfl_values = []
        for result in results:
            value = getattr(result, "localization_uncertainty", None)
            if value is not None:
                dfl_values.extend(np.asarray(value, dtype=float).reshape(-1).tolist())
        path = Path(self.experiment_dir) / "ddal_bcqs_selection_metrics.csv"
        if not path.exists():
            path.write_text(
                "round,processed_images,bcqs_total_ms,bcqs_ms_per_processed_image,"
                "average_labeled_objects,total_quota,quota_coverage,mean_dfl_entropy\n"
            )
        per_image_ms = 1000.0 * elapsed / processed_count if processed_count else 0.0
        mean_dfl = float(np.mean(dfl_values)) if dfl_values else float("nan")
        with path.open("a") as handle:
            handle.write(
                f"{int(self.round or 0)},{processed_count},{1000.0 * elapsed:.6f},"
                f"{per_image_ms:.6f},{average_objects:.8g},{quotas.sum():.8g},"
                f"{self._quota_coverage(achieved, quotas):.8g},{mean_dfl:.8g}\n"
            )

    def _write_artifacts(
        self,
        elapsed,
        processed_count,
        selected_indices,
        image_paths,
        results,
        processed_indices,
    ) -> None:
        if not self.experiment_dir:
            return
        experiment_dir = Path(self.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        time_log = experiment_dir / os.environ.get("TIME_LOGFILE", "time_log.csv")
        if not time_log.exists():
            time_log.write_text("Round,TotalTime,NumImages,TimePerImage\n")
        with time_log.open("a") as handle:
            time_per_image = elapsed / processed_count if processed_count else 0.0
            handle.write(
                f"{self.round},{elapsed:.4f},{processed_count},{time_per_image:.6f}\n"
            )
        selection_log = experiment_dir / os.environ.get(
            "SELECTION_LOGFILE", "selection_log.txt"
        )
        selected_paths = self._get_image_paths_for_indices(
            selected_indices, image_paths
        )
        with selection_log.open("a") as handle:
            handle.write(",".join(Path(path).name for path in selected_paths) + "\n")
        self._save_predictions_for_selection(
            experiment_dir=str(experiment_dir),
            round_num=self.round,
            selected_image_paths=selected_paths,
            image_paths=image_paths,
            selected_indices=selected_indices,
            results=results,
            unlabeled_indices=processed_indices,
        )
        self._save_selection_symlinks(str(experiment_dir), self.round, selected_paths)

    def get_strategy_name(self) -> str:
        return "ddal"
