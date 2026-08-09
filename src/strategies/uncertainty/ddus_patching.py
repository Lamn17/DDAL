from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou


QUALITY_MOMENTUM = 0.99
QUALITY_INITIAL_VALUE = 0.5
QUALITY_XI = 0.6
QUALITY_INITIAL_VALUES = None


def configure_quality_tracking(momentum=0.99, initial_value=0.5, xi=0.6, initial_values=None):
    global QUALITY_MOMENTUM, QUALITY_INITIAL_VALUE, QUALITY_XI, QUALITY_INITIAL_VALUES
    if not 0.0 <= momentum < 1.0:
        raise ValueError("quality momentum must be in [0, 1)")
    if not 0.0 <= initial_value <= 1.0:
        raise ValueError("quality initial value must be in [0, 1]")
    if not 0.0 <= xi <= 1.0:
        raise ValueError("quality xi must be in [0, 1]")
    QUALITY_MOMENTUM = float(momentum)
    QUALITY_INITIAL_VALUE = float(initial_value)
    QUALITY_XI = float(xi)
    if initial_values is None:
        QUALITY_INITIAL_VALUES = None
    else:
        QUALITY_INITIAL_VALUES = np.asarray(initial_values, dtype=np.float32).reshape(-1)


def ddus_detection_loss(self, preds, batch):
    """Ultralytics detection loss with detached DDUS quality capture."""
    parsed = self.parse_output(preds)
    assigned, loss, loss_detach = self.get_assigned_targets_and_loss(parsed, batch)
    fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
    pred_distri = parsed["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = parsed["scores"].permute(0, 2, 1).contiguous()
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri) * stride_tensor
    batch_size = pred_scores.shape[0]
    imgsz = (
        torch.tensor(parsed["feats"][0].shape[2:], device=self.device, dtype=pred_scores.dtype)
        * self.stride[0]
    )
    targets = torch.cat(
        (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
    )
    targets = self.preprocess(
        targets.to(self.device),
        batch_size,
        scale_tensor=imgsz[[1, 0, 1, 0]],
    )
    gt_labels = targets[..., :1]

    quality_rows = []
    with torch.no_grad():
        for batch_idx in range(batch_size):
            foreground = fg_mask[batch_idx]
            if foreground.sum() == 0:
                continue
            assigned_classes = gt_labels[
                batch_idx,
                target_gt_idx[batch_idx][foreground],
                0,
            ].long()
            pred_conf = pred_scores[batch_idx][foreground, assigned_classes].sigmoid()
            pred_box = pred_bboxes[batch_idx][foreground]
            target_box = target_bboxes[batch_idx][foreground]
            ious = bbox_iou(pred_box, target_box, xywh=False, CIoU=False).reshape(-1)
            combined = pred_conf.pow(QUALITY_XI) * ious.pow(1.0 - QUALITY_XI)
            quality_rows.append(
                torch.stack(
                    [assigned_classes.float(), combined, pred_conf, ious],
                    dim=1,
                )
            )

    self.ddus_batch_quality = (
        torch.cat(quality_rows, dim=0)
        if quality_rows
        else torch.zeros((0, 4), device=self.device)
    )
    return loss * batch_size, loss_detach


def update_classwise_quality_ema(
    tracker,
    class_quality_sum,
    class_cls_sum,
    class_loc_sum,
    class_quality_count,
    base_momentum=None,
):
    momentum = QUALITY_MOMENTUM if base_momentum is None else base_momentum
    observed = class_quality_count > 0
    if not observed.any():
        return

    for attribute, value_sum in (
        ("class_quality", class_quality_sum),
        ("class_cls_quality", class_cls_sum),
        ("class_loc_quality", class_loc_sum),
    ):
        current = getattr(tracker, attribute)
        averages = value_sum / class_quality_count.clamp_min(1.0)
        current[observed] = (
            momentum * current[observed] + (1.0 - momentum) * averages[observed]
        )


def _trainer_model(trainer):
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def _on_train_start(trainer):
    num_classes = trainer.model.nc if hasattr(trainer.model, "nc") else 80
    initial = QUALITY_INITIAL_VALUE
    class_quality = torch.full((num_classes,), initial, device=trainer.device)
    if QUALITY_INITIAL_VALUES is not None:
        previous = torch.as_tensor(QUALITY_INITIAL_VALUES, device=trainer.device)
        class_quality[: min(num_classes, len(previous))] = previous[:num_classes]
    trainer.class_quality = class_quality
    trainer.class_cls_quality = torch.full((num_classes,), initial, device=trainer.device)
    trainer.class_loc_quality = torch.full((num_classes,), initial, device=trainer.device)


def _on_train_batch_end(trainer):
    criterion = getattr(_trainer_model(trainer), "criterion", None)
    quality = getattr(criterion, "ddus_batch_quality", None)
    if quality is None or quality.numel() == 0:
        return

    num_classes = len(trainer.class_quality)
    classes = quality[:, 0].long()
    quality_sum = torch.zeros(num_classes, device=trainer.device)
    cls_sum = torch.zeros(num_classes, device=trainer.device)
    loc_sum = torch.zeros(num_classes, device=trainer.device)
    count = torch.zeros(num_classes, device=trainer.device)
    quality_sum.scatter_add_(0, classes, quality[:, 1])
    cls_sum.scatter_add_(0, classes, quality[:, 2])
    loc_sum.scatter_add_(0, classes, quality[:, 3])
    count.scatter_add_(0, classes, torch.ones_like(classes, dtype=torch.float32))
    update_classwise_quality_ema(
        trainer,
        quality_sum,
        cls_sum,
        loc_sum,
        count,
    )


def _on_train_epoch_end(trainer):
    model = _trainer_model(trainer)
    quality_values = {
        "classwise_quality": trainer.class_quality.detach().clone(),
        "classwise_cls_quality": trainer.class_cls_quality.detach().clone(),
        "classwise_loc_quality": trainer.class_loc_quality.detach().clone(),
    }
    for name, value in quality_values.items():
        setattr(model, name, value)
        np.save(Path(trainer.wdir) / f"{name}.npy", value.cpu().numpy())


def enable_class_quality_tracking(model):
    """Enable generic class-quality capture without replacing the trainer loop."""
    v8DetectionLoss.__call__ = ddus_detection_loss  # type: ignore
    yolo_model = model.model
    yolo_model.add_callback("on_train_start", _on_train_start)
    yolo_model.add_callback("on_train_batch_end", _on_train_batch_end)
    yolo_model.add_callback("on_train_epoch_end", _on_train_epoch_end)


def enable_ddus_tracking(model):
    """Backward-compatible DDUS entry point for class-quality tracking."""
    enable_class_quality_tracking(model)


def ensure_ddus_patching():
    """Compatibility entry point for callers that only need the loss patch."""
    v8DetectionLoss.__call__ = ddus_detection_loss  # type: ignore


def patching(model):
    """Compatibility entry point for the previous patching API."""
    enable_ddus_tracking(model)
    return model
