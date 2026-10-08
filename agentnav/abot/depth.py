"""Metric monocular depth inference with one-inference-per-frame caching."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from agentnav.abot.types import CameraIntrinsics


@dataclass(frozen=True)
class DepthPrediction:
    depth: np.ndarray
    confidence: np.ndarray | None = None


class DepthEstimator(Protocol):
    def predict(self, rgb: Any) -> DepthPrediction:
        ...


class Metric3DDepthEstimator:
    """Lazy Metric3D ViT-S wrapper using the validated raw metric scale."""

    def __init__(self, source: str, checkpoint: str, device: str = "cuda:0") -> None:
        self.source = source
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._torch = None
        self._cv2 = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import cv2
        import torch

        source = str(Path(self.source).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        import hubconf

        model = hubconf.metric3d_vit_small(pretrain=False)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        self._model = model.to(self.device).eval()
        self._torch = torch
        self._cv2 = cv2

    def predict(self, rgb: Any) -> DepthPrediction:
        self._load()
        assert self._model is not None and self._torch is not None and self._cv2 is not None
        torch = self._torch
        cv2 = self._cv2
        image = (
            np.asarray(rgb.convert("RGB"))
            if isinstance(rgb, Image.Image)
            else np.asarray(rgb, dtype=np.uint8)
        )
        input_h, input_w = 616, 1064
        height, width = image.shape[:2]
        scale = min(input_h / height, input_w / width)
        resized = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_h, pad_w = input_h - resized.shape[0], input_w - resized.shape[1]
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
        mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
        tensor = ((tensor - mean) / std)[None].to(self.device)
        with torch.inference_mode():
            pred = self._model.inference({"input": tensor})[0].squeeze()
        pred = pred[top : input_h - (pad_h - top), left : input_w - (pad_w - left)]
        pred = torch.nn.functional.interpolate(
            pred[None, None], (height, width), mode="bilinear"
        )[0, 0]
        # Metric3D canonical focal length is 1000 px.
        pred = pred * ((252.075 * scale) / 1000.0)
        depth = pred.clamp(0, 300).float().cpu().numpy().astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return DepthPrediction(depth=depth)


class UniDepthV2DepthEstimator:
    """Independent, optional near-field depth check using local UniDepth weights."""

    def __init__(
        self, source: str, checkpoint: str, camera: CameraIntrinsics,
        device: str = "cuda:0",
    ) -> None:
        self.source = source
        self.checkpoint = checkpoint
        self.camera = camera
        self.device = device
        self._model = None
        self._pinhole = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        source = str(Path(self.source).resolve())
        checkpoint = Path(self.checkpoint).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"UniDepth checkpoint directory missing: {checkpoint}")
        if source not in sys.path:
            sys.path.insert(0, source)
        from unidepth.models import UniDepthV2
        from unidepth.utils.camera import Pinhole

        self._model = UniDepthV2.from_pretrained(str(checkpoint)).to(self.device).eval()
        self._pinhole = Pinhole
        self._torch = torch

    def predict(self, rgb: Any) -> DepthPrediction:
        self._load()
        assert self._model is not None and self._torch is not None
        assert self._pinhole is not None
        torch = self._torch
        image = (
            np.asarray(rgb.convert("RGB"))
            if isinstance(rgb, Image.Image)
            else np.asarray(rgb, dtype=np.uint8)
        )
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).to(self.device)
        intrinsics = torch.tensor(
            [
                [self.camera.fx, 0.0, self.camera.cx],
                [0.0, self.camera.fy, self.camera.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.inference_mode():
            output = self._model.infer(tensor, self._pinhole(K=intrinsics))
        depth = output["depth"].squeeze().float().cpu().numpy().astype(np.float32)
        if depth.shape != image.shape[:2]:
            raise ValueError(f"UniDepth returned shape {depth.shape}, expected {image.shape[:2]}")
        depth[~np.isfinite(depth)] = 0.0
        return DepthPrediction(depth=depth)


class ObservationDepthCache:
    def __init__(self, estimator: DepthEstimator) -> None:
        self.estimator = estimator
        self._frame_id: int | None = None
        self._prediction: DepthPrediction | None = None
        self.inference_count = 0

    def reset(self) -> None:
        self._frame_id = None
        self._prediction = None
        self.inference_count = 0

    def get(self, frame_id: int, rgb: Any) -> DepthPrediction:
        if self._frame_id != int(frame_id) or self._prediction is None:
            self._prediction = self.estimator.predict(rgb)
            self._frame_id = int(frame_id)
            self.inference_count += 1
        return self._prediction
