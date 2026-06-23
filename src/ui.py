import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


RiskColor = Tuple[int, int, int]
Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


RISK_STYLES: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "color": (0, 180, 0),
        "text": "LOW",
        "warning": "Safe: low pedestrian risk",
    },
    "MEDIUM": {
        "color": (0, 165, 255),
        "text": "MEDIUM",
        "warning": "Caution: nearby moving object",
    },
    "HIGH": {
        "color": (0, 0, 255),
        "text": "HIGH",
        "warning": "Warning: high collision risk",
    },
}

DEFAULT_STYLE = {
    "color": (180, 180, 180),
    "text": "UNKNOWN",
    "warning": "No risk level available",
}


def get_risk_style(risk_level: Optional[str]) -> Dict[str, Any]:
    """Return color and text style for a risk level."""
    if not risk_level:
        return DEFAULT_STYLE

    return RISK_STYLES.get(str(risk_level).upper(), DEFAULT_STYLE)


def get_risk_level_from_score(score: Optional[float]) -> str:
    """Classify a risk score when a detection does not already include a level."""
    if score is None:
        return "UNKNOWN"
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    origin: Point,
    font_scale: float = 0.55,
    text_color: RiskColor = (255, 255, 255),
    background_color: RiskColor = (0, 0, 0),
    thickness: int = 1,
    padding: int = 5,
) -> None:
    """Draw readable text on a filled background."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x1 = max(x, 0)
    y1 = max(y - text_h - padding * 2, 0)
    x2 = min(x + text_w + padding * 2, image.shape[1] - 1)
    y2 = min(y + baseline, image.shape[0] - 1)

    cv2.rectangle(image, (x1, y1), (x2, y2), background_color, -1)
    cv2.putText(
        image,
        text,
        (x1 + padding, y2 - baseline - padding),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def normalize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize detection keys used by YOLO/risk modules.

    Accepted box formats:
    - bbox: [x1, y1, x2, y2]
    - box: [x1, y1, x2, y2]
    - x1, y1, x2, y2 keys
    """
    box = detection.get("bbox") or detection.get("box")
    if box is None:
        box = [
            detection.get("x1", 0),
            detection.get("y1", 0),
            detection.get("x2", 0),
            detection.get("y2", 0),
        ]

    x1, y1, x2, y2 = [int(value) for value in box]
    score = detection.get("risk_score", detection.get("score"))
    risk_level = detection.get("risk_level") or detection.get("level")
    risk_level = str(risk_level or get_risk_level_from_score(score)).upper()

    return {
        "class_name": detection.get("class_name", detection.get("class", "object")),
        "confidence": detection.get("confidence", detection.get("conf")),
        "risk_score": score,
        "risk_level": risk_level,
        "bbox": (x1, y1, x2, y2),
    }


def draw_detection_box(image: np.ndarray, detection: Dict[str, Any]) -> None:
    """Draw one object box, label, center point, and risk score."""
    normalized = normalize_detection(detection)
    x1, y1, x2, y2 = normalized["bbox"]
    style = get_risk_style(normalized["risk_level"])
    color = style["color"]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(image, (center_x, center_y), 5, color, -1)

    confidence = normalized["confidence"]
    confidence_text = "" if confidence is None else f" {float(confidence):.2f}"
    score = normalized["risk_score"]
    score_text = "" if score is None else f" | score {int(score)}"
    label = (
        f"{normalized['class_name']}{confidence_text} | "
        f"{normalized['risk_level']}{score_text}"
    )

    label_y = y1 - 8 if y1 > 30 else y2 + 24
    draw_text_with_background(
        image=image,
        text=label,
        origin=(x1, label_y),
        font_scale=0.55,
        background_color=color,
        thickness=1,
    )


def get_overall_risk(detections: Iterable[Dict[str, Any]]) -> Tuple[str, Optional[float]]:
    """Return the highest risk level and score from all detections."""
    normalized = [normalize_detection(item) for item in detections]
    if not normalized:
        return "LOW", None

    priority = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    highest = max(
        normalized,
        key=lambda item: (
            priority.get(item["risk_level"], 0),
            item["risk_score"] if item["risk_score"] is not None else -1,
        ),
    )
    return highest["risk_level"], highest["risk_score"]


def draw_warning_banner(
    image: np.ndarray,
    detections: Iterable[Dict[str, Any]],
    alpha: float = 0.82,
) -> None:
    """Draw a top warning banner for the overall scene risk."""
    detection_list = list(detections)
    overall_level, overall_score = get_overall_risk(detection_list)
    style = get_risk_style(overall_level)
    overlay = image.copy()

    banner_h = 58
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_h), style["color"], -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    score_text = "" if overall_score is None else f" | max score {int(overall_score)}"
    message = f"{style['warning']} | risk {style['text']}{score_text}"
    cv2.putText(
        image,
        message,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_summary_panel(image: np.ndarray, detections: Iterable[Dict[str, Any]]) -> None:
    """Draw object count and risk-level legend in the lower-left corner."""
    detection_list = [normalize_detection(item) for item in detections]
    panel_w = 260
    panel_h = 112
    margin = 12
    x1 = margin
    y1 = image.shape[0] - panel_h - margin
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
    for detection in detection_list:
        counts[detection["risk_level"]] = counts.get(detection["risk_level"], 0) + 1

    cv2.putText(
        image,
        f"Detected objects: {len(detection_list)}",
        (x1 + 12, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for index, level in enumerate(("HIGH", "MEDIUM", "LOW")):
        style = get_risk_style(level)
        row_y = y1 + 52 + index * 20
        cv2.rectangle(image, (x1 + 12, row_y - 10), (x1 + 28, row_y + 6), style["color"], -1)
        cv2.putText(
            image,
            f"{level}: {counts.get(level, 0)}",
            (x1 + 38, row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def visualize_results(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Draw all visualization elements and optionally save/show the result image.

    Each detection should include class/class_name, bbox or box, confidence,
    risk_score/score, and risk_level/level when available.
    """
    result_image = image.copy()

    for detection in detections:
        draw_detection_box(result_image, detection)

    draw_warning_banner(result_image, detections)
    draw_summary_panel(result_image, detections)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, result_image)

    if show:
        cv2.imshow("Pedestrian Risk AI", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image


def load_detections(json_path: str) -> List[Dict[str, Any]]:
    """Load detection results from a JSON file for standalone UI testing."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("detections", [])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw object detection and risk UI.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--detections", required=True, help="Detection JSON path")
    parser.add_argument("--output", default="outputs/ui_result.jpg", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show result window")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    detections = load_detections(args.detections)
    visualize_results(image, detections, output_path=args.output, show=args.show)
    print(f"Visualization saved: {args.output}")


if __name__ == "__main__":
    main()
