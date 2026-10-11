"""Visual overlay annotator for real-time detection bounding boxes, badges, and performance HUD."""

from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from yolo_hailo_mlops.deployment.postprocess import Detection


def generate_distinct_colors(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """Generate perceptually distinct, vibrant BGR colors using golden ratio hue spacing."""
    colors: List[Tuple[int, int, int]] = []
    golden_ratio = 0.618033988749895
    h = 0.1
    for _ in range(num_classes):
        h = (h + golden_ratio) % 1.0
        # Saturation: 0.85, Value: 0.95 for maximum clarity and vibrancy
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        # Convert to BGR integers [0, 255]
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


class VisualAnnotator:
    """Renders sleek bounding box overlays, class tags, and performance telemetry HUD."""

    def __init__(
        self,
        num_classes: int = 80,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        hud_enabled: bool = True,
    ) -> None:
        self.colors = generate_distinct_colors(max(num_classes, 80))
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.hud_enabled = hud_enabled

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        return self.colors[class_id % len(self.colors)]

    def draw_detection(self, img: np.ndarray, det: Detection) -> None:
        """Draw bounding box and high-contrast label pill badge."""
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = self.get_color(det.class_id)

        # 1. Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

        # Corner accents for a modern edge look
        corner_len = min(15, max(5, int((x2 - x1) * 0.2)))
        accent_thick = self.box_thickness + 1
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, accent_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, accent_thick, cv2.LINE_AA)

        # 2. Label pill badge
        label = f"{det.class_name} {det.confidence * 100.0:.1f}%"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, 1)

        # Place label above box if possible, or inside if near top border
        lbl_y2 = y1 - 4 if y1 - lbl_h - 8 > 0 else y1 + lbl_h + 8
        lbl_y1 = lbl_y2 - lbl_h - 4
        lbl_x1 = max(0, x1)
        lbl_x2 = lbl_x1 + lbl_w + 8

        # Draw filled pill background
        cv2.rectangle(img, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, cv2.FILLED)

        # High-contrast text (white text or dark text depending on luminance)
        text_color = (255, 255, 255)
        cv2.putText(
            img,
            label,
            (lbl_x1 + 4, lbl_y2 - 4),
            self.font,
            self.font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        img: np.ndarray,
        stats: Dict[str, Any],
    ) -> None:
        """Render modern heads-up telemetry display card in top-left corner."""
        fps = stats.get("fps", 0.0)
        infer_ms = stats.get("infer_latency_ms", 0.0)
        backend = stats.get("backend_name", "Hailo-8L")
        inference_device = stats.get("inference_device", "NPU (Hailo HAT)" if "Hailo" in backend else "CPU")
        num_dets = stats.get("num_detections", 0)
        source = stats.get("source_name", "live")

        lines = [
            f"Inference: {inference_device}",
            f"Engine: {backend}",
            f"FPS: {fps:.1f}  |  Latency: {infer_ms:.1f}ms",
            f"Source: {source}  |  Objects: {num_dets}",
        ]

        card_x, card_y = 12, 12
        line_height = 20
        card_w = 340
        card_h = len(lines) * line_height + 16

        # Draw translucent dark background card
        sub_img = img[card_y : card_y + card_h, card_x : card_x + card_w]
        if sub_img.shape[0] == card_h and sub_img.shape[1] == card_w:
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 1.0)
            img[card_y : card_y + card_h, card_x : card_x + card_w] = res

        # Draw border
        cv2.rectangle(
            img,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (60, 200, 100) if "Hailo" in backend else (200, 150, 50),
            1,
            cv2.LINE_AA,
        )

        # Draw text lines
        for i, text in enumerate(lines):
            y_pos = card_y + 18 + (i * line_height)
            color = (0, 255, 180) if i == 0 else (240, 240, 240)
            cv2.putText(
                img,
                text,
                (card_x + 10, y_pos),
                self.font,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Annotate a copy of the input frame with all detections and HUD overlay."""
        annotated = frame.copy()

        for det in detections:
            self.draw_detection(annotated, det)

        if self.hud_enabled and stats:
            self.draw_hud(annotated, stats)

        return annotated
