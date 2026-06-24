from ultralytics import YOLO
import cv2
import json
import os
import random

from risk_analysis import (
    calculate_risk,
    get_risk_level
)
from ui import visualize_results

# =========================
# 경로 설정
# =========================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DETECTIONS_JSON_PATH = os.path.join(OUTPUT_DIR, "detections.json")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def select_random_image():
    """test_images 폴더에서 입력 이미지를 지정"""
    image_files = [
        file_name for file_name in os.listdir(TEST_IMAGES_DIR)
        if file_name.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        return None

    selected_file = random.choice(image_files)
    return os.path.join(TEST_IMAGES_DIR, selected_file)


def make_output_image_path(IMAGE_PATH):
    """선택된 이미지 이름을 반영한 결과 이미지 경로 생성"""
    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    return os.path.join(OUTPUT_DIR, f"result_{image_name}.jpg")


def save_detections_json(IMAGE_PATH, detections):
    """객체 인식 결과와 위험도 결과를 JSON 파일로 저장"""
    data = {
        "image": IMAGE_PATH,
        "detections": detections
    }

    with open(DETECTIONS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("모델 파일을 찾을 수 없습니다:", MODEL_PATH)
        return

    if not os.path.isdir(TEST_IMAGES_DIR):
        print("test_images 폴더를 찾을 수 없습니다:", TEST_IMAGES_DIR)
        return

    IMAGE_PATH = select_random_image()
    if IMAGE_PATH is None:
        print("test_images 폴더에 사용할 수 있는 이미지가 없습니다.")
        return

    output_image_path = make_output_image_path(IMAGE_PATH)

    print("선택된 입력 이미지:")
    print(IMAGE_PATH)

    # 1. 모델 로드
    model = YOLO(MODEL_PATH)

    # 2. 이미지 로드
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("이미지를 불러올 수 없습니다:", IMAGE_PATH)
        return

    img_h, img_w = image.shape[:2]

    # 3. 예측 수행
    results = model.predict(
        source=IMAGE_PATH,
        conf=0.25,
        save=False
    )

    result = results[0]
    names = model.names

    print("\n===== 객체 검출 결과 =====")

    risk_results = []
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        print("검출된 객체가 없습니다.")
        save_detections_json(IMAGE_PATH, detections)
        visualize_results(
            image=image,
            detections=detections,
            output_path=output_image_path,
            show=True
        )

        print("\nJSON 저장 완료:")
        print(DETECTIONS_JSON_PATH)
        print("\n최종 결과 화면 저장 완료:")
        print(output_image_path)
        return

    # 4. 검출 결과 순회
    for idx, box_data in enumerate(result.boxes, start=1):
        cls_id = int(box_data.cls[0].item())
        conf = float(box_data.conf[0].item())
        class_name = names[cls_id]

        # 박스 좌표
        x1, y1, x2, y2 = box_data.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 객체 크기
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h

        # 객체 중심 좌표
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 위험도 계산
        prediction = {
            "class": class_name,
            "x": center_x,
            "width": box_w,
            "height": box_h
        }

        risk_score = calculate_risk(
            prediction,
            img_w,
            img_h
        )

        risk_level = get_risk_level(
            risk_score
        )

        risk_results.append({
            "class": class_name,
            "score": risk_score,
            "level": risk_level
        })

        detections.append({
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "width": box_w,
            "height": box_h,
            "area": box_area,
            "confidence": conf,
            "risk_score": risk_score,
            "risk_level": risk_level
        })

        # 콘솔 출력
        print(f"[{idx}] 객체 종류: {class_name}")
        print(f"    신뢰도: {conf:.2f}")
        print(f"    박스 좌표: ({x1}, {y1}, {x2}, {y2})")
        print(f"    중심 위치: ({center_x}, {center_y})")
        print(f"    크기: width={box_w}, height={box_h}, area={box_area}")
        print(f"    위험도 점수: {risk_score}")
        print(f"    위험도 등급: {risk_level}")

    # 가장 위험한 객체 계산
    most_dangerous = max(
        risk_results,
        key=lambda x: x["score"]
    )

    print("\n가장 위험한 객체")
    print(most_dangerous)

    # 5. 객체 인식 결과 및 위험도 결과 JSON 저장
    save_detections_json(IMAGE_PATH, detections)

    print("\nJSON 저장 완료:")
    print(DETECTIONS_JSON_PATH)

    # 6. UI 시각화 모듈을 통해 최종 결과 화면 저장 및 출력
    visualize_results(
        image=image,
        detections=detections,
        output_path=output_image_path,
        show=True
    )

    print("\n최종 결과 화면 저장 완료:")
    print(output_image_path)


if __name__ == "__main__":
    main()
