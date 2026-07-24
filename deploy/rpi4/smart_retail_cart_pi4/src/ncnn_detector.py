"""Lightweight YOLO NCNN/ONNX inference used by the Raspberry Pi package.

The exported models in this package have a single output shaped
``(4 + number_of_classes, number_of_predictions)``.  Keeping this code free
of Ultralytics and PyTorch makes the normal NCNN path usable on Raspberry Pi
OS 32-bit as well as 64-bit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class Detector(Protocol):
    names: list[str]

    def predict(self, frame: np.ndarray) -> list[Detection]: ...


def load_names(model_path: Path) -> list[str]:
    """Read class names from an Ultralytics NCNN metadata file when present."""
    metadata = (
        model_path / "metadata.yaml" if model_path.is_dir() else model_path.parent / "metadata.yaml"
    )
    if metadata.exists():
        names: dict[int, str] = {}
        in_names = False
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line == "names:":
                in_names = True
                continue
            if in_names:
                match = re.match(r"^\s+(\d+):\s*(.+?)\s*$", line)
                if match:
                    names[int(match.group(1))] = match.group(2).strip("'\"")
                    continue
                if line and not line.startswith((" ", "\t")):
                    break
        if names:
            return [names[index] for index in range(max(names) + 1)]
    return ["apple", "hao_hao_noodles", "orange", "sprite"]


def load_image_size(model_path: Path) -> int | None:
    """Read the square input size recorded in an NCNN metadata file."""
    metadata = model_path / "metadata.yaml" if model_path.is_dir() else model_path.parent / "metadata.yaml"
    if not metadata.exists():
        return None

    match = re.search(
        r"^imgsz:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)\s*$",
        metadata.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match and match.group(1) == match.group(2):
        return int(match.group(1))
    return None


def validate_settings(imgsz: int, confidence: float, iou: float) -> None:
    if imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer.")
    if not 0 < confidence <= 1:
        raise ValueError("--conf must be greater than 0 and at most 1.")
    if not 0 < iou <= 1:
        raise ValueError("--iou must be greater than 0 and at most 1.")


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize while preserving aspect ratio, matching Ultralytics preprocessing."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = size - resized_width
    pad_height = size - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


def preprocess(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    padded, scale, left, top = letterbox(frame, size)
    # NCNN/ONNX model input is NCHW RGB with pixels normalized to [0, 1].
    blob = padded[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(blob), scale, left, top


def postprocess(
    output: np.ndarray,
    frame_shape: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Convert exported YOLO output to image-space boxes and apply class-wise NMS."""
    predictions = np.asarray(output, dtype=np.float32)
    predictions = np.squeeze(predictions)
    if predictions.ndim != 2:
        raise RuntimeError(f"Unexpected model output shape: {predictions.shape}")
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.shape[1] < 5:
        raise RuntimeError(f"Unexpected model output shape: {predictions.shape}")

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]
    selected = confidences >= confidence_threshold

    height, width = frame_shape
    candidates: dict[int, tuple[list[list[float]], list[float], list[int]]] = {}
    for index in np.flatnonzero(selected):
        center_x, center_y, box_width, box_height = boxes[index]
        x1 = (center_x - box_width / 2 - pad_left) / scale
        y1 = (center_y - box_height / 2 - pad_top) / scale
        x2 = (center_x + box_width / 2 - pad_left) / scale
        y2 = (center_y + box_height / 2 - pad_top) / scale
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[index])
        item = candidates.setdefault(class_id, ([], [], []))
        item[0].append([x1, y1, x2 - x1, y2 - y1])
        item[1].append(float(confidences[index]))
        item[2].append(int(index))

    detections: list[Detection] = []
    for class_id, (class_boxes, class_confidences, _) in candidates.items():
        kept = cv2.dnn.NMSBoxes(class_boxes, class_confidences, confidence_threshold, iou_threshold)
        for kept_index in np.asarray(kept).reshape(-1):
            x, y, box_width, box_height = class_boxes[int(kept_index)]
            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=class_confidences[int(kept_index)],
                    x1=round(x),
                    y1=round(y),
                    x2=round(x + box_width),
                    y2=round(y + box_height),
                )
            )
    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


class NcnnDetector:
    def __init__(self, model: Union[str, Path], imgsz: int, confidence: float, iou: float, threads: int = 2):
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError(
                "NCNN runtime is not installed. Run: .venv/bin/python -m pip install -r requirements-rpi.txt"
            ) from exc

        self.model_path = Path(model)
        if not self.model_path.is_dir():
            raise RuntimeError("An NCNN model must be a directory ending in '_ncnn_model'.")
        parameter = self.model_path / "model.ncnn.param"
        binary = self.model_path / "model.ncnn.bin"
        if not parameter.is_file() or not binary.is_file():
            raise RuntimeError(f"NCNN files not found in: {self.model_path}")

        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.names = load_names(self.model_path)
        expected_size = load_image_size(self.model_path)
        if expected_size is not None and self.imgsz != expected_size:
            raise ValueError(
                f"Model {self.model_path.name} expects --imgsz {expected_size}, got {self.imgsz}."
            )
        self.net = ncnn.Net()
        self.net.opt.num_threads = max(1, threads)
        self.net.opt.use_vulkan_compute = False
        if self.net.load_param(str(parameter)) != 0 or self.net.load_model(str(binary)) != 0:
            raise RuntimeError(f"Could not load NCNN model from: {self.model_path}")

    def predict(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, left, top = preprocess(frame, self.imgsz)
        with self.net.create_extractor() as extractor:
            extractor.input("in0", self._ncnn_mat(blob))
            status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError(f"NCNN inference failed with status {status}")
        return postprocess(
            np.array(output), frame.shape[:2], scale, left, top, self.confidence, self.iou
        )

    @staticmethod
    def _ncnn_mat(blob: np.ndarray):
        import ncnn

        return ncnn.Mat(blob).clone()


class OnnxDetector:
    def __init__(self, model: Union[str, Path], imgsz: int, confidence: float, iou: float):
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is optional and not installed. Run: .venv/bin/python -m pip install -r requirements-rpi-onnx.txt"
            ) from exc

        self.model_path = Path(model)
        if not self.model_path.is_file():
            raise RuntimeError(f"ONNX model not found: {self.model_path}")
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.names = load_names(self.model_path)
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        input_metadata = self.session.get_inputs()[0]
        self.input_name = input_metadata.name
        input_shape = input_metadata.shape
        if (
            len(input_shape) == 4
            and isinstance(input_shape[2], int)
            and isinstance(input_shape[3], int)
        ):
            if input_shape[2] != input_shape[3]:
                raise RuntimeError(f"Only square ONNX inputs are supported, got shape {input_shape}.")
            if self.imgsz != input_shape[2]:
                raise ValueError(
                    f"Model {self.model_path.name} expects --imgsz {input_shape[2]}, got {self.imgsz}."
                )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, left, top = preprocess(frame, self.imgsz)
        output = self.session.run(None, {self.input_name: blob[None]})[0]
        return postprocess(output, frame.shape[:2], scale, left, top, self.confidence, self.iou)


def create_detector(
    model: Union[str, Path], imgsz: int, confidence: float, iou: float, threads: int = 2
) -> Detector:
    validate_settings(imgsz, confidence, iou)
    model_path = Path(model)
    if model_path.suffix.lower() == ".onnx":
        return OnnxDetector(model_path, imgsz, confidence, iou)
    return NcnnDetector(model_path, imgsz, confidence, iou, threads)


def annotate(frame: np.ndarray, detections: list[Detection], names: list[str]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        color = (0, 200, 0)
        cv2.rectangle(annotated, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        name = names[detection.class_id] if detection.class_id < len(names) else str(detection.class_id)
        label = f"{name} {detection.confidence:.2f}"
        text_y = max(detection.y1 - 8, 18)
        cv2.putText(annotated, label, (detection.x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated
