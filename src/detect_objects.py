from ultralytics import YOLO
import cv2
import os

from risk_analysis import (
    calculate_risk,
    get_risk_level
)

# =========================
# 경로 설정
# =========================
PROJECT_ROOT = r"C:\Users\ansl\OneDrive - 숙명여자대학교\바탕 화면\pedestrian-risk-ai"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
IMAGE_PATH = os.path.join(PROJECT_ROOT, "test_images", "test1.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
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

    output_img = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        print("검출된 객체가 없습니다.")
        return

    print("\n===== 객체 검출 결과 =====")

    risk_results = []

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
        "score": risk_score
        })

        # 콘솔 출력
        print(f"[{idx}] 객체 종류: {class_name}")
        print(f"    신뢰도: {conf:.2f}")
        print(f"    박스 좌표: ({x1}, {y1}, {x2}, {y2})")
        print(f"    중심 위치: ({center_x}, {center_y})")
        print(f"    크기: width={box_w}, height={box_h}, area={box_area}")

        print(f"    위험도 점수: {risk_score}")
        print(f"    위험도 등급: {risk_level}")

        # 이미지에 박스/라벨 그리기
        label = f"{class_name} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 중심점 표시
        cv2.circle(output_img, (center_x, center_y), 4, (0, 0, 255), -1)

        # 중심 좌표 텍스트
        center_text = f"({center_x},{center_y})"
        cv2.putText(output_img, center_text, (center_x + 5, center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    #가장 위험한 객체 계산
    most_dangerous = max(
    risk_results,
    key=lambda x: x["score"]
    )

    print("\n가장 위험한 객체")
    print(most_dangerous)

    # 5. 결과 이미지 저장
    output_path = os.path.join(OUTPUT_DIR, "result_test1.jpg")
    cv2.imwrite(output_path, output_img)

    print("\n결과 이미지 저장 완료:")
    print(output_path)

if __name__ == "__main__":
    main()