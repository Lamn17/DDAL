import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from ..base import BaseStrategy

sys.path.append(str(Path(__file__).resolve().parents[3]))
from scripts.score_maple_crops import CLIP_MEAN, CLIP_STD, load_maple_model


_MAPLE_MODEL_CACHE: Dict[Tuple, torch.nn.Module] = {}


def clear_maple_model_cache() -> None:
    """Release MaPLe models cached for repeated benchmark queries."""
    _MAPLE_MODEL_CACHE.clear()


class MaPLeUncertaintyStrategy(BaseStrategy):
    def __init__(
        self,
        model,
        maple_checkpoint: Optional[str] = None,
        maple_checkpoint_pattern: Optional[str] = None,
        maple_checkpoint_fallback_latest: bool = True,
        data_yaml: Optional[str] = None,
        maple_weight: float = 1.0,
        entropy_weight: float = 1.0,
        disagreement_weight: float = 1.0,
        inverse_confidence_weight: float = 0.0,
        yolo_confidence_weight: float = 0.0,
        padding: float = 0.15,
        batch_size: int = 16,
        device: str = "auto",
        maple_device: Optional[str] = None,
        n_ctx: int = 2,
        ctx_init: str = "",
        prompt_depth: int = 9,
        keep_checkpoint_tokens: bool = False,
        synthetic_maple: bool = False,
        cache_maple_model: bool = False,
        experiment_dir: Optional[str] = None,
        round: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.maple_checkpoint = maple_checkpoint
        self.maple_checkpoint_pattern = maple_checkpoint_pattern
        self.maple_checkpoint_fallback_latest = maple_checkpoint_fallback_latest
        self.data_yaml = data_yaml
        self.maple_weight = maple_weight
        self.entropy_weight = entropy_weight
        self.disagreement_weight = disagreement_weight
        self.inverse_confidence_weight = inverse_confidence_weight
        self.yolo_confidence_weight = yolo_confidence_weight
        self.padding = padding
        self.batch_size = batch_size
        self.device = device
        self.maple_device = maple_device
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.prompt_depth = prompt_depth
        self.keep_checkpoint_tokens = keep_checkpoint_tokens
        self.synthetic_maple = synthetic_maple
        self.cache_maple_model = cache_maple_model
        self.experiment_dir = experiment_dir
        self.round = round
        self.inference_device = kwargs.get("inference_device", device)
        self.inference_batch_size = kwargs.get("inference_batch_size", 1)
        self.class_names = self._load_class_names(data_yaml)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)
        start_time = time.time()
        local_kwargs = dict(kwargs)
        num_inf = local_kwargs.pop("num_inference", -1)
        candidate_indices = unlabeled_indices[:num_inf] if num_inf > 0 else unlabeled_indices
        candidate_paths = self._get_image_paths_for_indices(candidate_indices, image_paths)

        results = self.model.inference(
            candidate_paths,
            return_boxes=True,
            return_classes=True,
            return_probs=True,
            num_inference=-1,
            inference_device=self.inference_device,
            inference_batch_size=self.inference_batch_size,
            **local_kwargs,
        )

        torch_device = self._resolve_torch_device(self.maple_device or self.device)
        maple_checkpoint = None if self.synthetic_maple else self._resolve_maple_checkpoint()
        if maple_checkpoint:
            print(f"Using MaPLe checkpoint: {maple_checkpoint}")
        cache_key = (
            maple_checkpoint or "synthetic",
            tuple(self.class_names),
            str(torch_device),
            self.n_ctx,
            self.ctx_init,
            self.prompt_depth,
        )
        maple_model = _MAPLE_MODEL_CACHE.get(cache_key) if self.cache_maple_model else None
        if maple_model is None:
            maple_model = load_maple_model(
                SimpleNamespace(
                    backbone="ViT-B/16",
                    n_ctx=self.n_ctx,
                    ctx_init=self.ctx_init,
                    prompt_depth=self.prompt_depth,
                    checkpoint=maple_checkpoint,
                    keep_checkpoint_tokens=self.keep_checkpoint_tokens,
                ),
                self.class_names,
                torch_device,
            )
            if self.cache_maple_model:
                _MAPLE_MODEL_CACHE[cache_key] = maple_model

        image_rows: Dict[int, Dict] = {
            int(idx): {
                "image_index": int(idx),
                "image_path": path,
                "num_boxes": 0,
                "mean_entropy": 0.0,
                "max_entropy": 0.0,
                "mean_disagreement": 0.0,
                "mean_inverse_confidence": 0.0,
                "mean_yolo_confidence": 0.0,
                "maple_image_score": 0.0,
            }
            for idx, path in zip(candidate_indices, candidate_paths)
        }
        crop_rows: List[Dict] = []
        batch_tensors: List[torch.Tensor] = []
        batch_rows: List[Dict] = []

        for image_index, image_path, result in zip(candidate_indices, candidate_paths, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            classes = result.classes if result.classes is not None else np.zeros(len(result.boxes), dtype=int)
            confs = result.probs if result.probs is not None else np.ones(len(result.boxes), dtype=float)

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                img_w, img_h = image.size
                for obj_idx, (box, yolo_cls, yolo_conf) in enumerate(zip(result.boxes, classes, confs)):
                    crop_box = self._square_crop_box(
                        tuple(float(v) for v in box[:4]), img_w, img_h, self.padding
                    )
                    crop = image.crop(crop_box)
                    batch_tensors.append(self.transform(crop))
                    batch_rows.append(
                        {
                            "image_index": int(image_index),
                            "image_path": image_path,
                            "object_index": obj_idx,
                            "yolo_cls": int(yolo_cls),
                            "yolo_confidence": float(yolo_conf),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )
                    if len(batch_tensors) >= self.batch_size:
                        self._flush_batch(maple_model, batch_tensors, batch_rows, torch_device, crop_rows)
                        batch_tensors = []
                        batch_rows = []

        self._flush_batch(maple_model, batch_tensors, batch_rows, torch_device, crop_rows)
        self._aggregate_scores(crop_rows, image_rows)
        self._write_score_logs(image_rows, crop_rows)

        scores = np.array([image_rows[int(idx)]["maple_image_score"] for idx in candidate_indices])
        top_local = np.argsort(scores)[-n_samples:]
        selected_indices = candidate_indices[top_local]

        elapsed = time.time() - start_time
        print(
            f"MaPLe uncertainty selected {len(selected_indices)}/{len(candidate_indices)} candidates "
            f"in {elapsed:.2f}s"
        )
        print(
            f"MaPLe score statistics - Mean: {scores.mean():.4f}, Std: {scores.std():.4f}, "
            f"Min: {scores.min():.4f}, Max: {scores.max():.4f}"
        )
        return selected_indices

    def _flush_batch(self, maple_model, batch_tensors, batch_rows, device, crop_rows) -> None:
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            probs = maple_model(images).softmax(dim=-1).cpu()
        preds = probs.argmax(dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)

        for row, prob, pred, ent in zip(batch_rows, probs, preds, entropy):
            pred_id = int(pred)
            maple_conf = float(prob[pred_id])
            row["maple_pred_id"] = pred_id
            row["maple_pred_name"] = self.class_names[pred_id]
            row["maple_confidence"] = maple_conf
            row["maple_entropy"] = float(ent)
            row["maple_disagreement"] = int(int(row["yolo_cls"]) != pred_id)
            row["inverse_confidence"] = 1.0 - maple_conf
            crop_rows.append(row)

    def _aggregate_scores(self, crop_rows: List[Dict], image_rows: Dict[int, Dict]) -> None:
        by_image: Dict[int, List[Dict]] = {}
        for row in crop_rows:
            by_image.setdefault(int(row["image_index"]), []).append(row)

        max_entropy = max(1e-8, math.log(max(2, len(self.class_names))))
        for image_index, rows in by_image.items():
            entropies = np.array([float(row["maple_entropy"]) / max_entropy for row in rows])
            disagreements = np.array([float(row["maple_disagreement"]) for row in rows])
            inverse_conf = np.array([float(row["inverse_confidence"]) for row in rows])
            yolo_conf = np.array([float(row["yolo_confidence"]) for row in rows])
            score = (
                self.entropy_weight * float(entropies.mean())
                + self.disagreement_weight * float(disagreements.mean())
                + self.inverse_confidence_weight * float(inverse_conf.mean())
                + self.yolo_confidence_weight * float(1.0 - yolo_conf.mean())
            )
            row = image_rows[int(image_index)]
            row.update(
                {
                    "num_boxes": len(rows),
                    "mean_entropy": float(entropies.mean()),
                    "max_entropy": float(entropies.max()),
                    "mean_disagreement": float(disagreements.mean()),
                    "mean_inverse_confidence": float(inverse_conf.mean()),
                    "mean_yolo_confidence": float(yolo_conf.mean()),
                    "maple_image_score": self.maple_weight * score,
                }
            )

    def _write_score_logs(self, image_rows: Dict[int, Dict], crop_rows: List[Dict]) -> None:
        if not self.experiment_dir or self.round is None:
            return
        output_dir = Path(self.experiment_dir) / "maple_scores" / f"round_{self.round}"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_score_path = output_dir / "image_scores.csv"
        crop_score_path = output_dir / "crop_scores.csv"
        ranked_path = output_dir / "image_scores_ranked.csv"

        image_score_rows = list(image_rows.values())
        self._write_csv(image_score_path, image_score_rows)
        self._write_csv(crop_score_path, crop_rows)
        ranked = sorted(image_score_rows, key=lambda row: float(row["maple_image_score"]), reverse=True)
        self._write_csv(ranked_path, ranked)
        print(f"Saved MaPLe uncertainty scores to {output_dir}")

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _square_crop_box(
        xyxy: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
        padding: float,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = xyxy
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        side = max(box_w, box_h) * (1.0 + padding)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        crop_x1 = int(round(cx - side / 2.0))
        crop_y1 = int(round(cy - side / 2.0))
        crop_x2 = int(round(cx + side / 2.0))
        crop_y2 = int(round(cy + side / 2.0))
        crop_x1 = max(0, min(crop_x1, img_w - 1))
        crop_y1 = max(0, min(crop_y1, img_h - 1))
        crop_x2 = max(crop_x1 + 1, min(crop_x2, img_w))
        crop_y2 = max(crop_y1 + 1, min(crop_y2, img_h))
        return crop_x1, crop_y1, crop_x2, crop_y2

    @staticmethod
    def _resolve_torch_device(device_value: str) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        value = str(device_value or "auto")
        if value == "auto":
            return torch.device("cuda:0")
        if value.isdigit():
            return torch.device(f"cuda:{value}")
        return torch.device(value)

    def _resolve_maple_checkpoint(self) -> str:
        if self.maple_checkpoint_pattern:
            target_round = max(0, int(self.round or 0) - 1)
            checkpoint = Path(
                self.maple_checkpoint_pattern.format(
                    round=target_round,
                    selection_round=int(self.round or 0),
                    experiment_dir=self.experiment_dir or "",
                )
            )
            if checkpoint.exists():
                return str(checkpoint)

            if self.maple_checkpoint_fallback_latest:
                candidates = []
                for round_id in range(target_round - 1, -1, -1):
                    candidate = Path(
                        self.maple_checkpoint_pattern.format(
                            round=round_id,
                            selection_round=int(self.round or 0),
                            experiment_dir=self.experiment_dir or "",
                        )
                    )
                    if candidate.exists():
                        candidates.append(candidate)
                        break
                if candidates:
                    print(
                        f"MaPLe checkpoint for round {target_round:03d} not found; "
                        f"falling back to {candidates[0]}"
                    )
                    return str(candidates[0])

            raise FileNotFoundError(
                f"MaPLe checkpoint not found for round {target_round:03d}: {checkpoint}"
            )

        if self.maple_checkpoint:
            checkpoint = Path(self.maple_checkpoint)
            if checkpoint.exists():
                return str(checkpoint)
            raise FileNotFoundError(f"MaPLe checkpoint not found: {checkpoint}")

        raise ValueError("MaPLe strategy requires maple_checkpoint or maple_checkpoint_pattern")

    @staticmethod
    def _load_class_names(data_yaml: Optional[str]) -> List[str]:
        if not data_yaml:
            raise ValueError("MaPLe strategy requires data_yaml to load class names")
        with Path(data_yaml).open("r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        return [str(name) for name in names]

    def get_strategy_name(self) -> str:
        return "maple"
