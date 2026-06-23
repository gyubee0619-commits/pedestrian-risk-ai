import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np


Color = Tuple[int, int, int]
Box = Tuple[int, int, int, int]
Detection = Dict[str, Any]


RISK_STYLES: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "color": (34, 170, 76),
        "label": "LOW",
        "message": "LOW RISK - Keep monitoring",
    },
    "MEDIUM": {
        "color": (0, 165, 255),
        "label": "MEDIUM",
        "message": "CAUTION - Moving object nearby",
    },
    "HIGH": {
        "color": (0, 0, 230),
        "label": "HIGH",
        "message": "WARNING - High collision risk",
    },
}

DEFAULT_STYLE = {
    "color": (150, 150, 150),
    "label": "UNKNOWN",
    "message": "UNKNOWN RISK",
}


def get_risk_level_from_score(score: Optional[Union[int, float]]) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def get_risk_style(risk_level: Optional[str]) -> Dict[str, Any]:
    if risk_level is None:
        return DEFAULT_STYLE
    return RISK_STYLES.get(str(risk_level).upper(), DEFAULT_STYLE)


def normalize_box(detection: Detection) -> Box:
    box = detection.get("bbox") or detection.get("box")
    if box is not None:
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2), int(y2)

    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        return (
            int(detection["x1"]),
            int(detection["y1"]),
            int(detection["x2"]),
            int(detection["y2"]),
        )

    x = int(detection.get("x", 0))
    y = int(detection.get("y", 0))
    width = int(detection.get("width", 0))
    height = int(detection.get("height", 0))
    return x, y, x + width, y + height


def normalize_detection(detection: Detection) -> Detection:
    x1, y1, x2, y2 = normalize_box(detection)
    width = int(detection.get("width", x2 - x1))
    height = int(detection.get("height", y2 - y1))
    score = detection.get("risk_score", detection.get("score"))
    risk_level = detection.get("risk_level", detection.get("level"))
    risk_level = str(risk_level or get_risk_level_from_score(score)).upper()

    return {
        "class_name": detection.get(
            "class_name",
            detection.get("class", detection.get("object_type", "object")),
        ),
        "bbox": (x1, y1, x2, y2),
        "width": width,
        "height": height,
        "confidence": detection.get("confidence", detection.get("conf")),
        "risk_score": score,
        "risk_level": risk_level,
    }


def normalize_detections(detections: Iterable[Detection]) -> List[Detection]:
    return [normalize_detection(detection) for detection in detections]


def get_overall_risk(detections: Iterable[Detection]) -> Tuple[str, Optional[float]]:
    normalized = normalize_detections(detections)
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


def put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.55,
    color: Color = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Color,
    scale: float = 0.48,
) -> None:
    padding = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    x1 = max(0, min(x, image.shape[1] - text_w - padding * 2 - 1))
    y2 = max(text_h + padding * 2, y)
    y1 = max(0, y2 - text_h - padding * 2 - baseline)
    x2 = min(image.shape[1] - 1, x1 + text_w + padding * 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    put_text(image, text, (x1 + padding, y2 - padding - baseline), scale=scale)


def draw_scaled_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    scale: float,
) -> None:
    for detection in detections:
        item = normalize_detection(detection)
        x1, y1, x2, y2 = item["bbox"]
        x1, y1, x2, y2 = (
            int(x1 * scale),
            int(y1 * scale),
            int(x2 * scale),
            int(y2 * scale),
        )
        style = get_risk_style(item["risk_level"])
        color = style["color"]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(image, center, 5, color, -1)

        confidence = item["confidence"]
        confidence_text = "" if confidence is None else f" {float(confidence):.2f}"
        score = item["risk_score"]
        score_text = "" if score is None else f" / {int(score)}"
        label = f"{item['class_name']}{confidence_text} | {item['risk_level']}{score_text}"
        label_y = y1 - 7 if y1 > 28 else y2 + 25
        draw_label(image, label, x1, label_y, color)

# 상단 경고 패널
def draw_top_bar(
    canvas: np.ndarray,
    detections: Iterable[Detection],
    top_height: int,
) -> None:
    overall_level, overall_score = get_overall_risk(detections)
    style = get_risk_style(overall_level)
    color = style["color"]

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], top_height), color, -1)

    score_text = "" if overall_score is None else f" | max score {int(overall_score)}"
    put_text(canvas, "Pedestrian Risk AI", (20, 32), scale=0.85, thickness=2)
    put_text(
        canvas,
        f"{style['message']} | overall risk {style['label']}{score_text}",
        (20, 62),
        scale=0.58,
        thickness=1,
    )

# 우측 정보 패널
def draw_side_panel(
    canvas: np.ndarray,
    detections: Iterable[Detection],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
) -> None:
    normalized = normalize_detections(detections)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (32, 34, 38), -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (90, 90, 90), 1)

    put_text(canvas, "Result Summary", (panel_x + 18, panel_y + 34), scale=0.65, thickness=2)
    put_text(canvas, f"Detected objects: {len(normalized)}", (panel_x + 18, panel_y + 68), scale=0.52)

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    for item in normalized:
        counts[item["risk_level"]] = counts.get(item["risk_level"], 0) + 1

    y = panel_y + 105
    for level in ("HIGH", "MEDIUM", "LOW"):
        style = get_risk_style(level)
        cv2.rectangle(canvas, (panel_x + 18, y - 15), (panel_x + 38, y + 5), style["color"], -1)
        put_text(canvas, f"{level}: {counts[level]}", (panel_x + 50, y + 3), scale=0.5)
        y += 28

    y += 16
    put_text(canvas, "Objects", (panel_x + 18, y), scale=0.58, thickness=2)
    y += 28

    for idx, item in enumerate(normalized[:9], start=1):
        style = get_risk_style(item["risk_level"])
        score = item["risk_score"]
        score_text = "-" if score is None else str(int(score))
        confidence = item["confidence"]
        conf_text = "-" if confidence is None else f"{float(confidence):.2f}"
        line = f"{idx}. {item['class_name']} | {item['risk_level']} | {score_text} | {conf_text}"

        cv2.circle(canvas, (panel_x + 25, y - 5), 5, style["color"], -1)
        put_text(canvas, line, (panel_x + 40, y), scale=0.42)
        y += 24

        if y > panel_y + panel_h - 20:
            break


def render_final_screen(
    frame: np.ndarray,
    results: List[Detection],
    output_path: Optional[str] = None,
    show: bool = False,
    window_name: str = "Pedestrian Risk AI",
) -> np.ndarray:
    """
    입력 이미지와 분석 결과를 최종 UI 화면으로 만든다.

    화면은 상단 경고 영역, 중앙 이미지 영역, 우측 정보 패널로 분리해
    중요한 정보가 서로 가려지지 않도록 구성한다.
    """
    top_h = 82
    margin = 16
    panel_w = 340
    canvas_w = 1280
    canvas_h = 720
    content_h = canvas_h - top_h - margin * 2
    image_area_w = canvas_w - panel_w - margin * 3

    frame_h, frame_w = frame.shape[:2]
    scale = min(image_area_w / frame_w, content_h / frame_h)
    display_w = max(1, int(frame_w * scale))
    display_h = max(1, int(frame_h * scale))

    canvas = np.full((canvas_h, canvas_w, 3), (18, 20, 24), dtype=np.uint8)
    draw_top_bar(canvas, results, top_h)

    image_x = margin
    image_y = top_h + margin + (content_h - display_h) // 2
    panel_x = image_x + image_area_w + margin
    panel_y = top_h + margin

    cv2.rectangle(
        canvas,
        (image_x - 1, top_h + margin - 1),
        (image_x + image_area_w + 1, top_h + margin + content_h + 1),
        (70, 74, 82),
        1,
    )

    display_image = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    draw_scaled_detections(display_image, results, scale)
    canvas[image_y:image_y + display_h, image_x:image_x + display_w] = display_image

    draw_side_panel(canvas, results, panel_x, panel_y, panel_w, content_h)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, canvas)

    if show:
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas


def visualize_results(
    image: np.ndarray,
    detections: List[Detection],
    output_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    return render_final_screen(
        frame=image,
        results=detections,
        output_path=output_path,
        show=show,
    )


def visualize_image(
    image_path: str,
    results: List[Detection],
    output_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return render_final_screen(
        frame=image,
        results=results,
        output_path=output_path,
        show=show,
    )


def visualize_video(
    video_path: str,
    frame_results: Dict[int, List[Detection]],
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not load video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = 1280
    height = 720

    writer = None
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        results = frame_results.get(frame_index, [])
        result_frame = render_final_screen(frame, results)

        if writer is not None:
            writer.write(result_frame)

        if show:
            cv2.imshow("Pedestrian Risk AI", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def load_results(json_path: str) -> Union[List[Detection], Dict[int, List[Detection]]]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if "frames" in data:
        return {
            int(frame["frame_index"]): frame.get("detections", [])
            for frame in data["frames"]
        }

    return data.get("detections", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize object detection and risk analysis results."
    )
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--detections", required=True, help="Detection/risk result JSON path")
    parser.add_argument("--output", help="Output image or video path")
    parser.add_argument("--show", action="store_true", help="Show final result screen")
    args = parser.parse_args()

    results = load_results(args.detections)

    if args.image:
        if not isinstance(results, list):
            raise ValueError("Image visualization requires a detection list.")
        visualize_image(args.image, results, output_path=args.output, show=args.show)
        print("Final image screen generated.")
        return

    if args.video:
        if not isinstance(results, dict):
            raise ValueError("Video visualization requires frame-based detections.")
        visualize_video(args.video, results, output_path=args.output, show=args.show)
        print("Final video screen generated.")
        return

    raise ValueError("Either --image or --video must be provided.")


if __name__ == "__main__":
    main()
