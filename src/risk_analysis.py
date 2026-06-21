# 위험도 계산 함수

def calculate_risk(prediction, img_width, img_height):

    # 1. 객체 종류 점수
    type_score_table = {
        "car": 20,
        "motorcycle": 15,
        "bicycle": 10
    }

    type_score = type_score_table.get(
        prediction["class"], 0
    )

    # 2. 위치 점수

    x = prediction["x"]

    image_center = img_width / 2

    distance = abs(x - image_center)

    if distance <= img_width * 0.2:
        position_score = 40

    elif distance <= img_width * 0.4:
        position_score = 25

    else:
        position_score = 10

    # 3. 크기 점수

    box_area = (
        prediction["width"]
        * prediction["height"]
    )

    image_area = (
        img_width
        * img_height
    )

    area_ratio = box_area / image_area

    if area_ratio >= 0.15:
        size_score = 40

    elif area_ratio >= 0.05:
        size_score = 25

    else:
        size_score = 10

    # 4. 최종 위험도

    risk_score = ( type_score + position_score + size_score )

    return risk_score

# 위험도 등급 분류 함수

def get_risk_level(score):

    if score >= 80:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    else:
        return "LOW"