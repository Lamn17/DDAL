from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops
from ultralytics.utils.loss import v8DetectionLoss

from .base import BaseModel, InferenceResult


def _normalized_class_probabilities(
    class_scores: torch.Tensor, kept_indices: torch.Tensor
) -> np.ndarray:
    """Return normalized per-class scores for the candidates retained by NMS.

    Ultralytics detection heads emit independent sigmoid class scores, rather
    than a categorical softmax.  DCCU needs a categorical distribution to
    compute Shannon entropy, so normalize the retained candidate's class-score
    vector over classes.  ``kept_indices`` is supplied by Ultralytics NMS and
    indexes the pre-NMS candidate dimension.
    """
    if kept_indices.numel() == 0:
        return np.empty((0, class_scores.shape[0]), dtype=np.float32)

    selected = class_scores[:, kept_indices.long()].transpose(0, 1).clamp_min(0)
    normalizer = selected.sum(dim=1, keepdim=True)
    uniform = torch.full_like(selected, 1.0 / selected.shape[1])
    probabilities = torch.where(
        normalizer > torch.finfo(selected.dtype).eps,
        selected / normalizer.clamp_min(torch.finfo(selected.dtype).eps),
        uniform,
    )
    return probabilities.detach().cpu().numpy()


class _ClassProbabilityDetectionPredictor(DetectionPredictor):
    """Detection predictor that preserves pre-NMS class-score vectors.

    The stock ``Results.boxes`` API retains only ``(xyxy, confidence, class)``.
    This predictor runs the same NMS with ``return_idxs=True`` and attaches the
    original class scores of the retained candidate to each result as
    ``result.class_probs``.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        raw_predictions = preds[0] if isinstance(preds, (list, tuple)) else preds
        if (
            not isinstance(raw_predictions, torch.Tensor)
            or raw_predictions.ndim != 3
            or raw_predictions.shape[-1] == 6
            or getattr(self.model, "end2end", False)
        ):
            return super().postprocess(preds, img, orig_imgs, **kwargs)

        num_classes = len(getattr(self.model, "names", {}))
        if num_classes <= 0 or raw_predictions.shape[1] < 4 + num_classes:
            return super().postprocess(preds, img, orig_imgs, **kwargs)

        # Clone before NMS, which converts box coordinates in-place.
        class_scores = raw_predictions[:, 4:4 + num_classes, :].detach().clone()
        detections, kept_indices = nms.non_max_suppression(
            raw_predictions,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else num_classes,
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=True,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = self.construct_results(detections, img, orig_imgs, **kwargs)
        for result, scores, indices in zip(results, class_scores, kept_indices):
            result.class_probs = _normalized_class_probabilities(scores, indices)
        return results


class YOLOModel(BaseModel):
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_name: str = "yolo11n.pt",
                 feature_layers: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_name = model_name
        self.feature_layers = feature_layers or ["model.9", "model.12", "model.15", "model.18", "model.21"]
        
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = [] # type: ignore

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            if model_path:
                print(f"Warning: Model path {model_path} does not exist. Loading default model {model_name}.")
            self.model = YOLO(model_name)
            
        self.is_trained = model_path is not None
        
    def get_available_layers(self) -> List[str]:
        if not self.model:
            return []
        return [name for name, _ in self.model.model.named_modules() if not name.endswith(('.act', '.conv', '.bn'))] # type: ignore
        
    def set_feature_layers(self, layers: List[str]) -> None:
        self.feature_layers = layers
        
    def load(self, model_path: str) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = YOLO(model_path)
        self.is_trained = True
        
    def _register_hooks(self):
        self._remove_hooks()
        if not self.model: 
            print("Warning: Model not initialized, cannot register hooks.")
            return False
        
        model_dict = dict(self.model.model.named_modules()) # type: ignore
        found_hooks = 0
        
        for name in self.feature_layers:
            if name in model_dict:
                try:
                    hook = model_dict[name].register_forward_hook(self._create_hook(name))
                    self._hooks.append(hook)
                    found_hooks += 1
                except Exception:
                    pass
            else:
                print(f"Warning: Feature layer {name} not found in model.")
        
        return found_hooks > 0
        
    def _create_hook(self, name: str):
        def hook(module, input, output):
            data = None
            if isinstance(output, torch.Tensor):
                data = output
            elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                data = output[0]
            if data is not None:
                self._feature_maps[name] = data.detach().clone()
            else:
                print(f"Warning: Could not extract data from layer {name}")
        return hook

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_box_features(
        self, 
        boxes: np.ndarray, 
        image_size: tuple = (640, 640),
        feature_layer: str = "model.15"
    ) -> torch.Tensor:
        if boxes is None or len(boxes) == 0:
            return torch.zeros((0, 256))
        
        if feature_layer not in self._feature_maps:
            available = list(self._feature_maps.keys())
            if available:
                feature_layer = available[-1]
            else:
                return torch.zeros((len(boxes), 256))
        
        feature_map = self._feature_maps[feature_layer]
        if feature_map is None:
            return torch.zeros((len(boxes), 256))
        
        img_h, img_w = image_size
        device = feature_map.device
        
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
        cx = (boxes_t[:, 0] + boxes_t[:, 2]) / 2
        cy = (boxes_t[:, 1] + boxes_t[:, 3]) / 2
        cx_norm = (cx / img_w - 0.5) * 2
        cy_norm = (cy / img_h - 0.5) * 2
        
        grid = torch.stack([cx_norm, cy_norm], dim=-1).view(1, 1, -1, 2)
        
        sampled = F.grid_sample(
            feature_map, grid, 
            mode='bilinear', 
            align_corners=False,
            padding_mode='border'
        )
        
        per_box_features = sampled.squeeze(0).squeeze(1).T
        return per_box_features

    def forward_head_on_features(
        self, 
        features: torch.Tensor, 
        scale_idx: int = -1
    ) -> tuple:
        if self.model is None:
            return None, None
        
        if features is None or len(features) == 0:
            return None, None
        
        detect = self.model.model.model[-1]
        device = features.device
        
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)
        
        feat_channels = features.shape[1]
        if scale_idx == -1:
            channel_to_scale = {128: 0, 256: 1, 512: 2}
            scale_idx = channel_to_scale.get(feat_channels, 2)
        
        bbox_out = features.clone()
        for layer in detect.cv2[scale_idx]:
            bbox_out = layer(bbox_out)
        
        cls_out = features.clone()
        for layer in detect.cv3[scale_idx]:
            cls_out = layer(cls_out)
        
        N = features.shape[0]
        cls_output = cls_out.view(N, -1)
        
        bbox_raw = bbox_out.view(N, -1)
        reg_max = detect.reg_max
        if bbox_raw.shape[1] == 4 * reg_max:
            bbox_reshaped = bbox_raw.view(N, 4, reg_max)
            bbox_softmax = F.softmax(bbox_reshaped, dim=2)
            arange = torch.arange(reg_max, dtype=torch.float32, device=device)
            bbox_decoded = (bbox_softmax * arange).sum(dim=2)
        else:
            bbox_decoded = bbox_raw[:, :4] if bbox_raw.shape[1] >= 4 else bbox_raw
        
        return cls_output, bbox_decoded

    def get_regression_weight_vector(self) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(64)
        
        detect = self.model.model.model[-1]
        
        weights = []
        for cv2_branch in detect.cv2:
            final_conv = cv2_branch[-1]
            w = final_conv.weight.data
            w_summed = w.sum(dim=(2, 3)).flatten(1).mean(dim=0)
            weights.append(w_summed)
        
        avg_weight = torch.stack(weights).mean(dim=0)
        return avg_weight
        
    def _extract_features_from_feature_maps(self, image_shape: tuple) -> Optional[np.ndarray]:
        if not self._feature_maps:
            print(f"Warning: No feature maps available for extraction. Feature layers: {self.feature_layers}")
            return None
            
        combined_features = []
        img_h, img_w = image_shape[:2]
        
        for layer_name in self.feature_layers:
            if layer_name not in self._feature_maps:
                continue
                
            feature_map = self._feature_maps[layer_name]
            if feature_map is None or feature_map.numel() == 0:
                continue
                
            if feature_map.dim() == 4:
                pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()
            elif feature_map.dim() == 3:
                pooled = feature_map.mean(dim=-1)
            elif feature_map.dim() == 2:
                pooled = feature_map
            else:
                pooled = feature_map.flatten()
                
            if pooled.dim() == 0:
                pooled = pooled.unsqueeze(0)
            elif pooled.dim() > 1:
                pooled = pooled.flatten()
                
            pooled = F.normalize(pooled, p=2, dim=0)
            combined_features.append(pooled)
            
        if combined_features:
            final_vector = torch.cat(combined_features, dim=0)
            final_vector = F.normalize(final_vector, p=2, dim=0)
            return final_vector.cpu().numpy()
            
        return None
        
    def _load_image_for_gradient(self, image_path: str, imgsz: int = 640) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((imgsz, imgsz))
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t
        
    def _make_batch_from_image(self, image_path: str, imgsz: int = 640):
        img_t = self._load_image_for_gradient(image_path, imgsz)
        device = next(self.model.model.parameters()).device if self.model and self.model.model else torch.device('cpu') # type: ignore
        img = img_t.unsqueeze(0).to(device)
        
        batch = {
            "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
            "cls": torch.zeros(1, dtype=torch.long, device=device),  
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device),
        }
        return batch, img
    
    def _pool_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        if gradients.dim() == 4:
            return F.adaptive_avg_pool2d(gradients, (1, 1)).squeeze()
        elif gradients.dim() == 3:
            return F.adaptive_avg_pool2d(gradients.unsqueeze(0), (1, 1)).squeeze()
        elif gradients.dim() == 2:
            return F.adaptive_avg_pool2d(gradients.unsqueeze(0).unsqueeze(0), (1, 1)).squeeze()
        else:
            return gradients.mean() if gradients.numel() > 1 else gradients
    
    def _pool_layer_gradients(self, param_grads: List[torch.Tensor]) -> torch.Tensor:
        if not param_grads:
            return torch.tensor([])
        
        pooled_grads = []
        
        for param_grad in param_grads:
            if param_grad.numel() == 0:
                continue
            if param_grad.dim() == 4:
                pooled = F.adaptive_avg_pool2d(param_grad, (1, 1)).mean()
            elif param_grad.dim() == 2:
                pooled = param_grad.mean()
            elif param_grad.dim() == 1:
                pooled = param_grad.mean()
            else:
                pooled = param_grad.flatten().mean()
                
            pooled_grads.append(pooled.unsqueeze(0))
        
        if pooled_grads:
            result = torch.cat(pooled_grads)
            return result
        else:
            return torch.tensor([])

    def _compute_layer_gradients(self, image_path: str, inference_result, use_pool: bool = True) -> Optional[np.ndarray]:
        if self.model is None:
            print("Warning: Model not initialized")
            return None
            
        try:
            imgsz = 640
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
                imgsz = self.model.args.imgsz # type: ignore
                if isinstance(imgsz, (list, tuple)):
                    imgsz = imgsz[0]
            imgsz = int(imgsz) # type: ignore
            
            batch, img = self._make_batch_from_image(image_path, imgsz)
            
            original_mode = self.model.model.training # type: ignore
            
            try:
                self.model.model.training = True # type: ignore
                for param in self.model.model.parameters(): # type: ignore
                    param.requires_grad_(True)
                
                param_list = []
                model_dict = dict(self.model.model.named_modules()) # type: ignore
                
                for layer_name in self.feature_layers:
                    if layer_name in model_dict:
                        layer = model_dict[layer_name]
                        for param in layer.parameters():
                            if param.requires_grad:
                                param_list.append(param)
                
                if not param_list:
                    print("Warning: No trainable parameters found in feature layers")
                    return None
                    
                img_clone = img.clone().detach().requires_grad_(True)
                raw_outputs = self.model.model(img_clone) # type: ignore
                
                try:
                    loss_fn = v8DetectionLoss(self.model.model)
                    hyp = SimpleNamespace()
                    hyp.box = 7.5
                    hyp.cls = 0.5
                    hyp.dfl = 1.5
                    loss_fn.hyp = hyp
                    
                    loss, _ = loss_fn(raw_outputs, batch)
                    total_loss = loss.sum() if hasattr(loss, 'sum') else loss
                except Exception as e:
                    print(f"Warning: Failed to compute loss: {e}")
                    return None
                
                total_loss.backward()
                
                param_grads = []
                for p in param_list:
                    g = p.grad
                    if g is None:
                        param_grads.append(torch.zeros_like(p))
                    else:
                        param_grads.append(g.detach().clone())
                
                if param_grads:
                    if use_pool:
                        result = self._pool_layer_gradients(param_grads)
                        return result.cpu().numpy()
                    else:
                        result = torch.cat([p.flatten() for p in param_grads])
                        return result.cpu().numpy()
                
                return None
                
            finally:
                try:
                    if original_mode:
                        self.model.model.training = True # type: ignore
                    else:
                        self.model.model.eval() # type: ignore
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Failed to compute layer gradients for {image_path}: {e}")
            try:
                if hasattr(self.model, 'model') and self.model.model is not None:
                    self.model.model.eval() # type: ignore
            except:
                pass
            return None
    
    def _compute_feature_gradients(self, image_path: str, inference_result, use_pool: bool = True) -> Optional[np.ndarray]:
        if self.model is None:
            print("Warning: Model not initialized")
            return None

        try:
            imgsz = 640
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
                imgsz = self.model.args.imgsz # type: ignore
                if isinstance(imgsz, (list, tuple)):
                    imgsz = imgsz[0]
            imgsz = int(imgsz) # type: ignore
            batch, img = self._make_batch_from_image(image_path, imgsz)
            original_mode = self.model.model.training # type: ignore
            stored_features = {}
            hook_handles = []
            try:
                self.model.model.training = True # type: ignore
                for param in self.model.model.parameters(): # type: ignore
                    param.requires_grad_(True)
                model_dict = dict(self.model.model.named_modules()) # type: ignore
                def make_hook(layer_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            output.retain_grad()
                            stored_features[layer_name] = output
                        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                            output[0].retain_grad()
                            stored_features[layer_name] = output[0]
                    return hook
                for layer_name in self.feature_layers:
                    if layer_name in model_dict:
                        hook_handles.append(model_dict[layer_name].register_forward_hook(make_hook(layer_name)))
                img_clone = img.clone().detach().requires_grad_(True)
                raw_outputs = self.model.model(img_clone) # type: ignore
                loss_fn = v8DetectionLoss(self.model.model)
                hyp = SimpleNamespace()
                hyp.box, hyp.cls, hyp.dfl = 7.5, 0.5, 1.5
                loss_fn.hyp = hyp
                loss, _ = loss_fn(raw_outputs, batch)
                (loss.sum() if hasattr(loss, 'sum') else loss).backward()
                final_grads = []
                for layer_name in self.feature_layers:
                    feature = stored_features.get(layer_name)
                    if feature is None or feature.grad is None:
                        continue
                    gradient = feature.grad
                    if use_pool:
                        final_grads.append(self._pool_gradients(gradient).detach())
                    else:
                        final_grads.append(gradient.flatten().detach())
                return torch.cat(final_grads).cpu().numpy() if final_grads else None
            finally:
                for handle in hook_handles:
                    handle.remove()
                self.model.model.train(original_mode) # type: ignore
        except Exception as error:
            print(f"Warning: Failed to compute feature gradients for {image_path}: {error}")
            return None

    def uncertainty_directions(self, image_paths: List[str], **kwargs) -> List[np.ndarray]:
        """Return one differentiable pre-NMS entropy-sensitivity direction per image."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        imgsz = kwargs.pop("imgsz", None)
        if imgsz is None and hasattr(self.model, "args") and hasattr(self.model.args, "imgsz"):
            imgsz = self.model.args.imgsz
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        imgsz = int(imgsz or 640)
        model_core = self.model.model
        model_dict = dict(model_core.named_modules())
        original_mode = model_core.training
        directions: List[np.ndarray] = []

        try:
            model_core.eval()
            for image_path in image_paths:
                captured = {}
                handles = []

                def make_hook(layer_name):
                    def hook(module, inputs, output):
                        value = output if isinstance(output, torch.Tensor) else output[0] if output else None
                        if isinstance(value, torch.Tensor):
                            captured[layer_name] = value
                    return hook

                try:
                    for layer_name in self.feature_layers:
                        layer = model_dict.get(layer_name)
                        if layer is not None:
                            handles.append(layer.register_forward_hook(make_hook(layer_name)))
                    _, image = self._make_batch_from_image(image_path, imgsz)
                    with torch.enable_grad():
                        raw_outputs = model_core(image)
                        raw_feature_maps = self._raw_detection_feature_maps(raw_outputs)
                        uncertainty = self._pre_nms_class_entropy(raw_feature_maps)
                        if uncertainty is None or not captured:
                            raise RuntimeError("No differentiable detection scores or feature maps were available")
                        ordered_features = [captured[name] for name in self.feature_layers if name in captured]
                        gradients = torch.autograd.grad(
                            uncertainty,
                            ordered_features,
                            allow_unused=True,
                            retain_graph=False,
                        )
                    pooled = []
                    for feature, gradient in zip(ordered_features, gradients):
                        if gradient is None:
                            gradient = torch.zeros_like(feature)
                        if gradient.dim() == 4:
                            value = F.adaptive_avg_pool2d(gradient, (1, 1)).flatten()
                        elif gradient.dim() == 3:
                            value = gradient.mean(dim=-1).flatten()
                        else:
                            value = gradient.flatten()
                        pooled.append(value.detach())
                    if not pooled:
                        raise RuntimeError("No feature-map gradients were produced")
                    direction = torch.cat(pooled)
                    direction = F.normalize(direction, p=2, dim=0)
                    directions.append(direction.cpu().numpy())
                finally:
                    for handle in handles:
                        handle.remove()
        finally:
            model_core.train(original_mode)
        return directions

    def _raw_detection_feature_maps(self, outputs):
        """Extract raw detection-head maps across Ultralytics train/eval returns."""
        if isinstance(outputs, tuple):
            for value in reversed(outputs):
                if isinstance(value, (list, tuple)):
                    return [item for item in value if isinstance(item, torch.Tensor)]
        if isinstance(outputs, (list, tuple)):
            return [item for item in outputs if isinstance(item, torch.Tensor)]
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        return []

    def _pre_nms_class_entropy(self, feature_maps) -> Optional[torch.Tensor]:
        if not feature_maps or self.model is None:
            return None
        detect = self.model.model.model[-1]
        class_start = int(getattr(detect, "reg_max", 16)) * 4
        terms = []
        for feature_map in feature_maps:
            if feature_map.dim() < 3 or feature_map.shape[1] <= class_start:
                continue
            probabilities = feature_map[:, class_start:].sigmoid().clamp(1e-6, 1.0 - 1e-6)
            entropy = -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())
            terms.append(entropy.mean())
        return torch.stack(terms).mean() if terms else None

    def save(self, save_path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
            
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save'):
            self.model.save(save_path)
        
    def inference(self,
                  image_paths: List[str],
                  return_boxes: bool = True,
                  return_classes: bool = True,
                  return_logits: bool = False,
                  return_probs: bool = False,
                  return_class_probs: bool = False,
                  return_features: bool = False,
                  return_embeddings: bool = False,
                  return_gradients: bool = False,
                  gradient_type: str = "layer",
                  use_pool: bool = True,
                  num_inference: int = -1,
                  conf: float = 0.25,
                  iou: float = 0.7,
                  **kwargs) -> List[InferenceResult]:

        embedding_or_feature = return_embeddings or return_features
        return_embeddings = embedding_or_feature
        return_features = embedding_or_feature
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        if 'feature_layers' in kwargs:
            feature_layers = kwargs.pop('feature_layers')
            if isinstance(feature_layers, list):
                self.set_feature_layers(feature_layers)

        inference_batch_size = int(kwargs.pop("inference_batch_size", 1) or 1)
        inference_device = kwargs.pop("inference_device", kwargs.get("device", None))
        if isinstance(inference_device, list):
            inference_device = ",".join(str(d) for d in inference_device)
        if inference_device is not None and str(inference_device) != "auto":
            kwargs["device"] = inference_device
        
        inference_results = []
        num_inf = num_inference if num_inference > 0 else len(image_paths)
        image_paths = image_paths[:num_inf]
        predictor = _ClassProbabilityDetectionPredictor if return_class_probs else None
        using_probability_predictor = isinstance(
            getattr(self.model, "predictor", None), _ClassProbabilityDetectionPredictor
        )
        if return_class_probs != using_probability_predictor:
            # Ultralytics reuses its predictor instance. Reset it when changing
            # output contracts so ``predictor=...`` is honoured on this call.
            self.model.predictor = None
        if return_features:
            hooks_registered = self._register_hooks()
            if not hooks_registered:
                return_features = False
            
        def convert_result(result) -> InferenceResult:
            boxes = None
            if return_boxes and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()

            classes = None
            if return_classes and result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)

            probs = None
            if return_probs and result.boxes is not None:
                probs = result.boxes.conf.cpu().numpy()

            class_probs = None
            if return_class_probs:
                value = getattr(result, "class_probs", None)
                if value is None:
                    raise RuntimeError(
                        "YOLO did not expose pre-NMS class scores for the post-NMS detections"
                    )
                class_probs = np.asarray(value, dtype=float)
                if result.boxes is None or class_probs.shape != (len(result.boxes), len(self.model.model.names)):
                    raise RuntimeError("YOLO class probabilities do not align with post-NMS detections")

            features = None
            if return_features:
                if hasattr(result, 'orig_shape'):
                    features = self._extract_features_from_feature_maps(result.orig_shape)
                else:
                    features = self._extract_features_from_feature_maps((640, 640))

            return InferenceResult(
                boxes=boxes,
                classes=classes,
                logits=None,
                probs=probs,
                class_probs=class_probs,
                features=features,
                embeddings=features
            )

        try:
            use_batched_predict = inference_batch_size > 1 and not return_features and not return_gradients
            if use_batched_predict:
                for start in range(0, len(image_paths), inference_batch_size):
                    batch_paths = image_paths[start:start + inference_batch_size]
                    results = self.model(
                        batch_paths,
                        conf=conf,
                        iou=iou,
                        batch=inference_batch_size,
                        verbose=False,
                        predictor=predictor,
                        **kwargs
                    )
                    for image_path, result in zip(batch_paths, results):
                        if result is None:
                            print(f"Warning: No result for image {image_path}")
                            continue
                        inference_results.append(convert_result(result))
                return inference_results

            for i, image_path in enumerate(image_paths):
                self._feature_maps = {}
                
                results = self.model(image_path, 
                                   conf=conf, 
                                   iou=iou,
                                   verbose=False,
                                   predictor=predictor,
                                   **kwargs)
                
                result = results[0] if results else None
                if result is None:
                    print(f"Warning: No result for image {image_path}")
                    continue
                
                layer_gradients = None
                embedding_gradients = None
                
                if return_logits:
                    pass

                inference_result = convert_result(result)
                
                if return_gradients:
                    if gradient_type == "layer":
                        layer_gradients = self._compute_layer_gradients(image_path, inference_result, use_pool)
                    elif gradient_type == "feature":
                        embedding_gradients = self._compute_feature_gradients(image_path, inference_result, use_pool)
                    else:
                        print(f"Warning: Unknown gradient type '{gradient_type}'. Expected 'layer' or 'feature'.")
                    
                    inference_result.layer_gradients = layer_gradients
                    inference_result.embedding_gradients = embedding_gradients
                
                inference_results.append(inference_result)
                
        finally:
            if return_features:
                self._remove_hooks()
            
        return inference_results
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              save_dir: str = "runs/train",
              **kwargs) -> 'YOLOModel':
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            val=True,
            **kwargs
        )
        self.metrics = results
        run_dir = Path(getattr(results, "save_dir", save_dir))
        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        if best_path.exists():
            model_path = best_path
        elif last_path.exists():
            model_path = last_path
        else:
            model_path = best_path
        print(f"Training completed. Best model path: {model_path}")
        self.model_path = str(model_path)
        self.is_trained = True
        
        return self
    
    def val(self,
            data_yaml: str,
            batch_size: int = 32,
            imgsz: int = 640,
            save_dir: str = "runs/val",
            **kwargs) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            project=save_dir,
            name="exp",
            **kwargs
        )
        
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        else:
            if hasattr(results, 'box'):
                metrics['map50-95'] = results.box.map
                metrics['map50'] = results.box.map50
                metrics['precision'] = results.box.p.mean()
                metrics['recall'] = results.box.r.mean()
                
        return metrics
