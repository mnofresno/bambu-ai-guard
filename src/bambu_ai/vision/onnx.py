"""ONNX YOLO provider (onnxruntime) with CoreML/CPU/MPS execution providers.

Works with YOLOv8/YOLO11 ONNX exports (output [1, 4+num_classes, N] or
[1, N, 4+num_classes]). Failure classes map to guard signals via a label map
(built-in or config `vision.label_map` JSON file).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..models import Detection, DetectionContext, DetectionResult, Frame
from .base import VisionModel

log = logging.getLogger(__name__)

DEFAULT_LABEL_MAP: dict[str, str] = {
    "spaghetti": "spaghetti",
    "blob": "blob",
    "failure": "spaghetti",
    "adhesion_loss": "adhesion_loss",
    "collapse": "collapse",
    "air_printing": "air_printing",
    "object": "object",
    "part": "object",
    "bottle": "object",
    "cup": "object",
    "vase": "object",
}

VALID_FAILURE_SIGNALS = (
    "spaghetti", "blob", "adhesion_loss", "collapse", "air_printing", "object",
)

# COCO class names (80) for pretrained weights; failure weights override via label_map.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class OnnxYoloModel(VisionModel):
    def __init__(
        self,
        model_path: str | Path,
        backend: str = "coreml",
        input_size: int = 640,
        confidence: float = 0.35,
        label_map: dict[str, str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self.input_size = input_size
        self.confidence = confidence
        self.label_map = {**DEFAULT_LABEL_MAP, **(label_map or {})}
        self.name = f"onnx:{self.model_path.name}:{backend}"
        self._session = None
        self._input_name = "images"
        self._class_names: list[str] = []
        self._num_classes = 80

    def _execution_providers(self) -> list[tuple[str, dict]]:
        eps: list[tuple[str, dict]] = []
        if self.backend == "coreml":
            # onnxruntime>=1.19 CoreML EP no longer takes compute_units here
            eps.append(("CoreMLExecutionProvider", {}))
        elif self.backend == "mps":
            eps.append(("MPSExecutionProvider", {}))
        eps.append(("CPUExecutionProvider", {}))
        return eps

    def _load(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self.model_path), providers=self._execution_providers()
        )
        self._input_name = self._session.get_inputs()[0].name
        self._num_classes = self._infer_num_classes()
        if self._num_classes == len(COCO_NAMES):
            self._class_names = COCO_NAMES
        log.info(
            "onnx model loaded: %s (providers=%s, input=%s, classes=%s)",
            self.model_path.name, self._session.get_providers(),
            self.input_size, self._num_classes,
        )

    def _infer_num_classes(self) -> int:
        out = self._session.get_outputs()[0]
        dims = [d for d in out.shape if isinstance(d, int) and d > 0]
        for d in dims:
            if d >= 5:
                return d - 4
        return 80

    async def close(self) -> None:
        self._session = None

    # -- inference -----------------------------------------------------------

    async def analyze(self, frame: Frame, context: DetectionContext) -> DetectionResult:
        if self._session is None:
            self._load()
        t0 = time.perf_counter()
        blob = self._preprocess(self._decode(frame))
        outputs = self._session.run(None, {self._input_name: blob})
        detections = self._nms(self._postprocess(outputs[0]))
        dets = [Detection(label=l, confidence=c, bbox=b) for l, c, b in detections]
        result = DetectionResult(
            detections=dets,
            signal_scores=self._signals(dets),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return result

    def _decode(self, frame: Frame) -> np.ndarray:
        if frame.bgr is not None and isinstance(frame.bgr, np.ndarray):
            return frame.bgr
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(frame.data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        size = self.input_size
        from PIL import Image

        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = np.array(
            Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        )
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        blob = canvas.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
        return blob

    def _postprocess(self, raw: np.ndarray) -> list[tuple[str, float, tuple]]:
        raw = np.squeeze(raw)
        if raw.shape[0] >= raw.shape[1]:
            raw = raw.T  # normalize to [4+C, N]
        boxes, scores = raw[:4], raw[4:]
        det: list[tuple[str, float, tuple]] = []
        for i in range(scores.shape[1]):
            cls_id = int(np.argmax(scores[:, i]))
            conf = float(scores[cls_id, i])
            if conf < self.confidence:
                continue
            cx, cy, bw, bh = (float(v) for v in boxes[:, i])
            det.append((
                self._class_names[cls_id] if cls_id < len(self._class_names) else f"cls_{cls_id}",
                conf,
                (cx - bw / 2, cy - bh / 2, bw, bh),
            ))
        return det

    def _signals(self, dets: list[Detection]) -> dict[str, float]:
        signals: dict[str, float] = {s: 0.0 for s in VALID_FAILURE_SIGNALS}
        for d in dets:
            signal = self.label_map.get(d.label.lower())
            if signal in signals:
                signals[signal] = max(signals[signal], d.confidence)
        return signals

    def _nms(self, dets: list[tuple[str, float, tuple]], iou_threshold: float = 0.5) -> list[tuple[str, float, tuple]]:
        """Greedy per-class NMS to drop duplicate boxes."""
        by_class: dict[str, list[tuple[str, float, tuple]]] = {}
        for d in dets:
            by_class.setdefault(d[0], []).append(d)
        kept: list[tuple[str, float, tuple]] = []
        for _, items in by_class.items():
            items.sort(key=lambda d: d[1], reverse=True)
            for d in items:
                if all(self._iou(d[2], k[2]) < iou_threshold for k in kept if k[0] == d[0]):
                    kept.append(d)
        return kept

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
