"""YOLO11 post-processing, DFL coordinate decoding, letterboxing, and Non-Maximum Suppression (NMS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# Standard COCO 80 class labels for default inference
DEFAULT_COCO_CLASSES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}


@dataclass
class Detection:
    """Detected bounding box with confidence and class label."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in original image space
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


@dataclass
class LetterboxResult:
    """Letterbox transformed image and inverse coordinate transform parameters."""
    image: np.ndarray  # Padded resized image
    ratio: float       # Scale ratio (r)
    pad_w: float       # Horizontal padding on each side
    pad_h: float       # Vertical padding on each side
    orig_shape: Tuple[int, int]  # (height, width)


def letterbox(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
) -> LetterboxResult:
    """Resize and pad image while preserving aspect ratio."""
    h_orig, w_orig = image.shape[:2]
    target_h, target_w = target_shape

    # Calculate scale factor
    r = min(target_w / w_orig, target_h / h_orig)
    if not scale_up:
        r = min(r, 1.0)

    # Compute unpadded new dimensions
    new_unpad_w = int(round(w_orig * r))
    new_unpad_h = int(round(h_orig * r))

    # Compute padding
    dw = (target_w - new_unpad_w) / 2.0
    dh = (target_h - new_unpad_h) / 2.0

    # Resize image
    if (w_orig, h_orig) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    # Compute integer padding borders
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Pad with border
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return LetterboxResult(
        image=padded,
        ratio=r,
        pad_w=float(left),
        pad_h=float(top),
        orig_shape=(h_orig, w_orig),
    )


class YOLODecoder:
    """Decodes raw YOLO11 tensor outputs from Hailo-8L accelerator or ONNX models."""

    def __init__(
        self,
        num_classes: int = 80,
        class_names: Optional[Dict[int, str]] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_COCO_CLASSES
        self.strides = strides
        self.reg_max = reg_max
        self._dfl_weights = np.arange(reg_max, dtype=np.float32)

    def decode_hailo_heads(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Decode the 6 detection head tensors output by Hailo-8L HEF graph.

        Returns array of shape (N_anchors, 4 + num_classes) where:
        columns 0:4 are [x1, y1, x2, y2] in 640x640 letterbox coordinates,
        columns 4: are class probabilities.
        """
        # Match tensors by shape: (1, H, W, C) or (H, W, C)
        tensors_by_res: Dict[int, Dict[str, np.ndarray]] = {}

        for name, arr in outputs.items():
            s = arr.shape
            if len(s) == 4 and s[0] == 1:
                h, w, c = s[1], s[2], s[3]
                data = arr[0]
            elif len(s) == 3:
                h, w, c = s[0], s[1], s[2]
                data = arr
            elif len(s) == 4 and s[1] in (64, self.num_classes):
                # NCHW format
                data = arr[0].transpose(1, 2, 0)
                h, w, c = data.shape
            else:
                continue

            if h not in tensors_by_res:
                tensors_by_res[h] = {}

            if c == 4 * self.reg_max or c == 64:
                tensors_by_res[h]["box"] = data
            elif c == self.num_classes or c > 1:
                tensors_by_res[h]["cls"] = data

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        # Sort strides ascending: 8 (H=80), 16 (H=40), 32 (H=20)
        expected_resolutions = [640 // s for s in self.strides]

        for stride, exp_res in zip(self.strides, expected_resolutions):
            if exp_res not in tensors_by_res:
                continue
            res_dict = tensors_by_res[exp_res]
            if "box" not in res_dict or "cls" not in res_dict:
                continue

            box_nhwc = res_dict["box"]  # (H, W, 64)
            cls_nhwc = res_dict["cls"]  # (H, W, nc)
            h, w = box_nhwc.shape[:2]

            # 1. Class scores: numerically stable sigmoid
            cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_nhwc, -20.0, 20.0)))
            cls_scores = cls_scores.reshape(-1, self.num_classes)
            all_scores.append(cls_scores)

            # 2. Box regression: DFL softmax + expectation
            box_reshaped = box_nhwc.reshape(-1, 4, self.reg_max)  # (N_anchors, 4, 16)
            # Softmax along last dimension
            max_val = np.max(box_reshaped, axis=-1, keepdims=True)
            exp_val = np.exp(box_reshaped - max_val)
            softmax = exp_val / np.sum(exp_val, axis=-1, keepdims=True)
            # Expected distance: sum(softmax * [0..15])
            dist = np.sum(softmax * self._dfl_weights, axis=-1)  # (N_anchors, 4): l, t, r, b

            # 3. Anchor points grid
            xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            anchor_points = np.stack([xv + 0.5, yv + 0.5], axis=-1).reshape(-1, 2)

            lt = dist[:, :2]  # left, top
            rb = dist[:, 2:]  # right, bottom

            x1y1 = (anchor_points - lt) * float(stride)
            x2y2 = (anchor_points + rb) * float(stride)

            xyxy = np.concatenate([x1y1, x2y2], axis=-1)  # (N_anchors, 4)
            all_boxes.append(xyxy)

        if not all_boxes or not all_scores:
            return np.empty((0, 4 + self.num_classes), dtype=np.float32)

        concat_boxes = np.concatenate(all_boxes, axis=0)
        concat_scores = np.concatenate(all_scores, axis=0)
        return np.concatenate([concat_boxes, concat_scores], axis=1)

    def decode_onnx_output(self, output_arr: np.ndarray) -> np.ndarray:
        """Decode standard ONNX tensor of shape (1, 4 + nc, 8400) or (1, 8400, 4 + nc).

        Returns array of shape (N_anchors, 4 + nc) where columns 0:4 are [x1, y1, x2, y2].
        """
        arr = output_arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # Shape is either (84, N) or (N, 84)
        if arr.shape[0] == 4 + self.num_classes and arr.shape[1] != 4 + self.num_classes:
            arr = arr.T  # -> (N, 84)

        cx = arr[:, 0]
        cy = arr[:, 1]
        w = arr[:, 2]
        h = arr[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = arr[:, 4:]
        return np.concatenate([boxes, scores], axis=1)

    def postprocess(
        self,
        raw_preds: np.ndarray,
        letterbox_info: LetterboxResult,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """Convert decoded predictions to scaled, filtered, and NMS-suppressed Detections."""
        if raw_preds.size == 0 or len(raw_preds) == 0:
            return []

        boxes = raw_preds[:, :4]
        scores = raw_preds[:, 4:]

        # Max class confidence per candidate box
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= conf_threshold
        if not np.any(mask):
            return []

        cand_boxes = boxes[mask]
        cand_scores = max_scores[mask]
        cand_class_ids = class_ids[mask]

        # Invert letterbox transform back to original image coordinates
        r = letterbox_info.ratio
        pad_w = letterbox_info.pad_w
        pad_h = letterbox_info.pad_h
        orig_h, orig_w = letterbox_info.orig_shape

        # Scale coordinates
        x1 = (cand_boxes[:, 0] - pad_w) / r
        y1 = (cand_boxes[:, 1] - pad_h) / r
        x2 = (cand_boxes[:, 2] - pad_w) / r
        y2 = (cand_boxes[:, 3] - pad_h) / r

        # Clip to original image boundaries
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Convert to cv2.dnn.NMSBoxes format: [x, y, w, h] in integers
        w_box = np.maximum(0.0, x2 - x1)
        h_box = np.maximum(0.0, y2 - y1)
        cv_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in zip(x1, y1, w_box, h_box)]
        cv_scores = cand_scores.tolist()

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=cv_scores,
            score_threshold=float(conf_threshold),
            nms_threshold=float(iou_threshold),
        )

        detections: List[Detection] = []
        if len(indices) > 0:
            flat_indices = np.array(indices).flatten()
            for idx in flat_indices:
                cid = int(cand_class_ids[idx])
                cname = self.class_names.get(cid, f"class_{cid}")
                detections.append(
                    Detection(
                        bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                        confidence=float(cand_scores[idx]),
                        class_id=cid,
                        class_name=cname,
                    )
                )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
