from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..base import BaseStrategy


class ReCALStrategy(BaseStrategy):

    def __init__(
        self,
        model,
        experiment_dir: Optional[str] = None,
        round: Optional[int] = None,
        labeled_indices: Optional[Sequence[int]] = None,
        knn_k: int = 10,
        sampling_conf: float = 0.25,
        quality_initial: float = 0.5,
        use_dccu: bool = True,
        use_rnu: bool = True,
        seed: int = 42,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        if knn_k <= 0:
            raise ValueError("knn_k must be positive")
        if not 0.0 <= sampling_conf <= 1.0:
            raise ValueError("sampling_conf must be in [0, 1]")
        if not 0.0 <= quality_initial <= 1.0:
            raise ValueError("quality_initial must be in [0, 1]")
        for name, value in {"use_dccu": use_dccu, "use_rnu": use_rnu}.items():
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")

        self.experiment_dir = experiment_dir
        self.round = round
        self.labeled_indices = np.asarray(labeled_indices if labeled_indices is not None else [], dtype=int)
        self.knn_k = int(knn_k)
        self.sampling_conf = float(sampling_conf)
        self.quality_initial = float(quality_initial)
        self.use_dccu = bool(use_dccu)
        self.use_rnu = bool(use_rnu)
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self._warned_probability_fallback = False

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        local_kwargs = dict(kwargs)
        num_inference = int(local_kwargs.pop("num_inference", -1))
        candidate_indices = unlabeled_indices[:num_inference] if num_inference > 0 else unlabeled_indices
        if len(candidate_indices) < n_samples:
            raise ValueError(
                f"ReCAL received {len(candidate_indices)} inferred images for a budget of {n_samples}. "
                "Increase num_inference or reduce n_samples."
            )
        candidate_paths = self._get_image_paths_for_indices(candidate_indices, image_paths)

        start_time = time.time()
        results = self._inference(candidate_paths, local_kwargs)
        processed_count = min(len(results), len(candidate_indices))
        if processed_count < n_samples:
            raise RuntimeError(f"ReCAL only received {processed_count} inference results for budget {n_samples}")
        processed_indices = candidate_indices[:processed_count]
        results = results[:processed_count]

        candidate_features = self._feature_matrix(results)
        num_classes = self._num_classes(results)
        class_quality = self._load_class_quality(num_classes)
        dccu = np.asarray(
            [self._dccu_uncertainty(result, class_quality) for result in results], dtype=float
        )
        if not self.use_dccu:
            dccu = np.asarray([self._entropy_uncertainty(result) for result in results], dtype=float)

        labeled_indices = self._usable_labeled_indices(image_paths, processed_indices)
        labeled_paths = self._get_image_paths_for_indices(labeled_indices, image_paths)
        labeled_results = self._inference(labeled_paths, local_kwargs) if len(labeled_paths) else []
        labeled_features = self._feature_matrix(labeled_results, candidate_features.shape[1])
        rnu = (
            self._residual_neighbor_uncertainty(candidate_features, labeled_features, dccu)
            if self.use_rnu
            else np.zeros_like(dccu)
        )
        uncertainty = dccu + rnu
        selected_local = self._weighted_kmeans_select(candidate_features, uncertainty, n_samples)
        selected_indices = processed_indices[np.asarray(selected_local, dtype=int)]

        self._write_artifacts(
            elapsed=time.time() - start_time,
            processed_count=processed_count,
            selected_indices=selected_indices,
            image_paths=image_paths,
            results=results,
            processed_indices=processed_indices,
        )
        logger.info(
            "ReCAL selected {} images from {} candidates (mean {} {:.4f}, mean RNU {:.4f}).",
            len(selected_indices),
            processed_count,
            "DCCU" if self.use_dccu else "entropy",
            float(dccu.mean()) if dccu.size else 0.0,
            float(rnu.mean()) if rnu.size else 0.0,
        )
        return selected_indices

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
            return_features=True,
            num_inference=-1,
            conf=self.sampling_conf,
            inference_device=self.strategy_params.get(
                "inference_device", self.strategy_params.get("device", "auto")
            ),
            inference_batch_size=self.strategy_params.get("inference_batch_size", 1),
            **local_kwargs,
        )

    def _dccu_uncertainty(self, result, class_quality: np.ndarray) -> float:
        """Difficulty-Calibrated Composite Uncertainty from class-quality-weighted entropy."""
        probabilities = self._class_probabilities(result, len(class_quality))
        if probabilities is None:
            entropy, classes = self._binary_confidence_entropy(result)
        else:
            entropy = self._normalized_posterior_uncertainty(probabilities)
            classes = np.argmax(probabilities, axis=1)
        if not len(entropy):
            return 0.0
        classes = np.clip(classes, 0, len(class_quality) - 1)
        values = entropy * (1.0 - class_quality[classes])
        return float(np.sqrt(np.mean(np.square(values))))

    def _entropy_uncertainty(self, result) -> float:
        probabilities = self._class_probabilities(result, self._result_num_classes(result))
        entropy = (
            self._normalized_posterior_uncertainty(probabilities)
            if probabilities is not None
            else self._binary_confidence_entropy(result)[0]
        )
        return float(np.sqrt(np.mean(np.square(entropy))))

    def _class_probabilities(self, result, num_classes: int) -> Optional[np.ndarray]:
        explicit = getattr(result, "class_probs", None)
        if explicit is not None:
            probabilities = np.asarray(explicit, dtype=float)
        elif getattr(result, "logits", None) is not None:
            logits = np.asarray(result.logits, dtype=float)
            if logits.ndim != 2:
                raise ValueError("ReCAL logits must have shape (num_boxes, num_classes)")
            logits = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
        else:
            if not self._warned_probability_fallback:
                logger.warning(
                    "ReCAL did not receive per-box class probabilities; falling back to binary confidence uncertainty. "
                    "Expose InferenceResult.class_probs for exact posterior DCCU."
                )
                self._warned_probability_fallback = True
            return None
        if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
            raise ValueError("ReCAL class probabilities must have shape (num_boxes, num_classes)")
        if not len(probabilities):
            return np.empty((0, num_classes), dtype=float)
        probabilities = np.clip(probabilities, self.epsilon, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _binary_confidence_entropy(self, result) -> tuple[np.ndarray, np.ndarray]:
        confidence_values = getattr(result, "probs", None)
        confidences = np.asarray(
            confidence_values if confidence_values is not None else [], dtype=float
        ).reshape(-1)
        if not len(confidences):
            return np.empty(0, dtype=float), np.empty(0, dtype=int)
        classes = getattr(result, "classes", None)
        classes = np.asarray(
            classes if classes is not None else np.zeros(len(confidences), dtype=int), dtype=int
        ).reshape(-1)
        count = min(len(confidences), len(classes))
        confidences = np.clip(confidences[:count], self.epsilon, 1.0 - self.epsilon)
        entropy = -(
            confidences * np.log(confidences)
            + (1.0 - confidences) * np.log(1.0 - confidences)
        ) / np.log(2.0)
        return entropy, classes[:count]

    def _result_num_classes(self, result) -> int:
        explicit = getattr(result, "class_probs", None)
        if explicit is not None and np.asarray(explicit).ndim == 2:
            return max(1, np.asarray(explicit).shape[1])
        logits = getattr(result, "logits", None)
        if logits is not None and np.asarray(logits).ndim == 2:
            return max(1, np.asarray(logits).shape[1])
        return self._num_classes([result])

    def _normalized_posterior_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        if probabilities.shape[1] <= 1:
            return np.zeros(len(probabilities), dtype=float)
        return -np.sum(probabilities * np.log(probabilities + self.epsilon), axis=1) / np.log(
            probabilities.shape[1]
        )

    def _residual_neighbor_uncertainty(
        self,
        candidate_features: np.ndarray,
        labeled_features: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        count = len(candidate_features)
        if count < 2:
            return np.zeros(count, dtype=float)
        neighbors = NearestNeighbors(
            n_neighbors=min(self.knn_k + 1, count), metric="euclidean"
        ).fit(candidate_features)
        distances, indices = neighbors.kneighbors(candidate_features, return_distance=True)
        labeled_coverage = self._labeled_feature_coverage(candidate_features, labeled_features)
        values = np.zeros(count, dtype=float)
        for index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            keep = row_indices != index
            row_distances, row_indices = row_distances[keep], row_indices[keep]
            if not len(row_indices):
                continue
            affinity = np.exp(-np.square(row_distances))
            residual = np.maximum(affinity - labeled_coverage[row_indices], 0.0)
            values[index] = float(np.mean(residual * uncertainties[row_indices]))
        return values

    def _labeled_feature_coverage(
        self, candidate_features: np.ndarray, labeled_features: np.ndarray
    ) -> np.ndarray:
        if not len(labeled_features):
            return np.zeros(len(candidate_features), dtype=float)
        coverage = np.zeros(len(candidate_features), dtype=float)
        for start in range(0, len(labeled_features), 64):
            reference = labeled_features[start:start + 64]
            squared_distance = np.square(
                candidate_features[:, None, :] - reference[None, :, :]
            ).sum(axis=2)
            coverage = np.maximum(coverage, np.exp(-squared_distance).max(axis=1))
        return coverage

    def _weighted_kmeans_select(
        self, features: np.ndarray, uncertainties: np.ndarray, budget: int
    ) -> List[int]:
        if len(features) < budget:
            raise ValueError(f"Cannot form {budget} clusters from {len(features)} candidates")
        weights = np.clip(np.asarray(uncertainties, dtype=float), 0.0, None) + self.epsilon
        weights /= weights.sum()
        kmeans = KMeans(n_clusters=budget, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(features, sample_weight=weights)
        selected = []
        for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
            members = np.flatnonzero(labels == cluster_id)
            if not len(members):
                continue
            distances = np.square(features[members] - centroid).sum(axis=1)
            selected.append(int(members[np.argmin(distances)]))
        if len(selected) < budget:
            remaining = sorted(
                set(range(len(features))) - set(selected),
                key=lambda index: (-uncertainties[index], index),
            )
            selected.extend(remaining[:budget - len(selected)])
        return selected

    def _num_classes(self, results: Iterable) -> int:
        model_names = getattr(getattr(self.model, "model", None), "names", None)
        if isinstance(model_names, (dict, list, tuple)):
            return max(1, len(model_names))
        largest_class = -1
        for result in results:
            classes = getattr(result, "classes", None)
            if classes is not None and len(classes):
                largest_class = max(largest_class, int(np.max(classes)))
        return max(1, largest_class + 1)

    def _load_class_quality(self, num_classes: int) -> np.ndarray:
        value = getattr(getattr(self.model, "model", None), "classwise_quality", None)
        if value is not None:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return self._fit_quality_vector(value, num_classes)

        search_paths = []
        model_path = getattr(self.model, "model_path", None)
        if model_path:
            search_paths.append(Path(model_path).parent / "classwise_quality.npy")
        if self.experiment_dir and self.round is not None and self.round > 0:
            train_dir = Path(self.experiment_dir) / f"round_{self.round - 1}" / "train"
            search_paths.extend(train_dir.glob("*/weights/classwise_quality.npy"))
        for path in dict.fromkeys(search_paths):
            if path.exists():
                logger.info("ReCAL loaded classwise_quality from {}", path)
                return self._fit_quality_vector(np.load(path), num_classes)
        return np.full(num_classes, self.quality_initial, dtype=float)

    def _fit_quality_vector(self, value, num_classes: int) -> np.ndarray:
        quality = np.asarray(value, dtype=float).reshape(-1)
        fitted = np.full(num_classes, self.quality_initial, dtype=float)
        fitted[: min(num_classes, len(quality))] = quality[:num_classes]
        return np.clip(fitted, 0.0, 1.0)

    def _feature_matrix(self, results: Sequence, expected_dimension: Optional[int] = None) -> np.ndarray:
        vectors = []
        dimension = expected_dimension
        for result in results:
            value = getattr(result, "features", None)
            if value is None:
                value = getattr(result, "embeddings", None)
            if value is not None:
                vector = np.asarray(value, dtype=float).reshape(-1)
                if vector.size:
                    dimension = vector.size if dimension is None else dimension
                    vectors.append(vector)
                    continue
            vectors.append(None)
        dimension = dimension or 4
        completed = []
        for result, vector in zip(results, vectors):
            if vector is None:
                completed.append(self._box_geometry_feature(result, dimension))
            elif vector.size == dimension:
                completed.append(vector)
            elif vector.size > dimension:
                completed.append(vector[:dimension])
            else:
                completed.append(np.pad(vector, (0, dimension - vector.size)))
        matrix = np.asarray(completed, dtype=float) if completed else np.empty((0, dimension))
        return self._normalize_features(matrix)

    @staticmethod
    def _box_geometry_feature(result, dimension: int) -> np.ndarray:
        boxes = getattr(result, "boxes", None)
        base = np.zeros(4, dtype=float)
        if boxes is not None and len(boxes):
            boxes = np.asarray(boxes, dtype=float)
            widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            base = np.array([len(boxes), np.mean(widths * heights), np.std(widths), np.std(heights)])
        return base[:dimension] if dimension <= 4 else np.pad(base, (0, dimension - 4))

    def _usable_labeled_indices(self, image_paths, candidate_indices) -> np.ndarray:
        labeled = self.labeled_indices
        if not len(labeled) and self.experiment_dir and self.round and self.round > 0:
            metadata_path = Path(self.experiment_dir) / f"round_{self.round - 1}" / "metadata.yaml"
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
                    if 0 <= int(index) < len(image_paths) and int(index) not in candidate_set
                }
            ),
            dtype=int,
        )

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        if not len(features):
            return features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms

    def _write_artifacts(self, elapsed, processed_count, selected_indices, image_paths, results, processed_indices) -> None:
        if not self.experiment_dir:
            return
        experiment_dir = Path(self.experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        time_log = experiment_dir / os.environ.get("TIME_LOGFILE", "time_log.csv")
        if not time_log.exists():
            time_log.write_text("Round,TotalTime,NumImages,TimePerImage\n")
        with time_log.open("a") as handle:
            time_per_image = elapsed / processed_count if processed_count else 0.0
            handle.write(f"{self.round},{elapsed:.4f},{processed_count},{time_per_image:.6f}\n")
        selection_log = experiment_dir / os.environ.get("SELECTION_LOGFILE", "selection_log.txt")
        selected_paths = self._get_image_paths_for_indices(selected_indices, image_paths)
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
        return "recal"
