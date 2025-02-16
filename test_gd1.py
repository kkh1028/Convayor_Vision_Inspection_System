import time
import serial
import requests
import numpy
from io import BytesIO
import cv2
import random
from requests.auth import HTTPBasicAuth
import pandas as pd
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

def display_results_table_in_new_window(results):
    """결과를 pandas DataFrame으로 별도 창에 출력"""
    if results is None or 'objects' not in results or not results['objects']:
        print("No valid objects to display.")
        return

    data = []
    priority_order = ["RASPBERRY PICO", "bootsel", "CHIPSET", "oscillator", "USB", "Hole"]

    for obj in results['objects']:
        if obj is not None and 'class' in obj and 'score' in obj:
            if obj['score'] <= 0.5:
                continue  # Score가 0.5 이하인 경우 건너뜀
            class_name = obj['class'].replace("bootsel_narrowly", "bootsel")  # 클래스 이름 수정
            data.append({
                'Class': class_name,
                'Score (%)': f"{obj['score'] * 100:.1f}%"
            })

    df = pd.DataFrame(data)
    df['Priority'] = df['Class'].apply(lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    df.sort_values(by='Priority', inplace=True)
    df.drop(columns=['Priority'], inplace=True)

    # pandas DataFrame 출력
    print("\n=== Detection Results ===")
    print(df.to_string(index=False))
    return df

def get_img():
    """USB 카메라로부터 이미지를 가져오는 함수"""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera Error")
        exit(-1)
    ret, img = cam.read()
    cam.release()
    return img

def crop_img(img, size_dict):
    """이미지를 지정된 크기로 자르는 함수"""
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img

def inference_request(img: numpy.array):
    """이미지를 API에 전송하여 추론 요청을 하는 함수"""
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = BytesIO(img_encoded.tobytes())
    try:
        response = requests.post(api_url, auth=auth, headers={'Content-Type': 'image/jpeg'}, data=img_bytes)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print("Failed to connect to the API.")
        print(f"Error sending request: {e}")
        return None

def get_color_for_class(class_name):
    """클래스 이름에 따라 고정 색상을 반환"""
    colors = {
        "RASPBERRY PICO": (192, 192, 192),  # 회색
        "bootsel": (0, 255, 0),  # 초록색
        "CHIPSET": (255, 255, 0),  # 노란색
        "oscillator": (255, 165, 0),  # 주황색
        "USB": (255, 0, 0),  # 빨간색
        "Hole": (128, 0, 128),  # 보라색
    }
    return colors.get(class_name, (255, 255, 255))  # 기본값: 흰색

def draw_boxes(img, results):
    """API 응답에 따라 이미지에 경계 상자와 클래스 이름을 표시하는 함수"""
    if results is None or 'objects' not in results or not results['objects']:
        print("No valid objects found in results.")
        return

    low_score_classes = []
    required_classes = ["RASPBERRY PICO", "bootsel", "CHIPSET", "oscillator", "USB"]
    detected_classes = set()

    for obj in results['objects']:
        if obj is None or 'class' not in obj or 'box' not in obj:
            continue
        if obj['score'] <= 0.5:
            continue  # Score가 0.5 이하인 경우 건너뜀
        box = obj['box']
        class_name = obj['class'].replace("bootsel_narrowly", "bootsel")  # 클래스 이름 수정
        detected_classes.add(class_name)  # 탐지된 클래스 저장
        score = obj['score']

        if class_name != "Hole" and score <= 0.6:
            low_score_classes.append(f"{class_name}: {score * 100:.1f}%")

        color = get_color_for_class(class_name)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)  # 두께 유지
        label = f"{class_name[:3]}: {score * 100:.1f}%"
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # 글자 크기 감소

    # Low score 클래스 표시
    y_offset = img.shape[0] - 40  # 하단 여백 수정
    for idx, low_class in enumerate(low_score_classes):
        cv2.putText(img, low_class, (10, y_offset - idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 탐지되지 않은 클래스 표시
    missing_classes = [cls for cls in required_classes if cls not in detected_classes]
    if missing_classes:
        missing_text = f"Missing: {', '.join(missing_classes)}"
        cv2.putText(img, missing_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 빨간색으로 표시

    # 상태 표시 추가
    defect_status = analyze_defects(results)
    color = (0, 255, 0) if defect_status == "PASS" else (255, 165, 0) if defect_status == "Inspection" else (0, 0, 255)
    cv2.putText(img, defect_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 상태 표시

def analyze_defects(results):
    """결과 분석 후 상태 메시지 반환"""
    required_classes = ["RASPBERRY PICO", "bootsel", "CHIPSET", "oscillator", "USB"]
    hole_count = 0
    class_counts = {}

    for obj in results['objects']:
        if obj['score'] <= 0.5:
            continue  # Score가 0.5 이하인 경우 무시
        class_name = obj['class'].replace("bootsel_narrowly", "bootsel")  # 클래스 이름 수정
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if class_name == "Hole":
            hole_count += 1

    missing_classes = [cls for cls in required_classes if class_counts.get(cls, 0) == 0]
    if len(missing_classes) == 0 and hole_count >= 3:
        return "PASS"
    elif len(missing_classes) >= 2:
        return "FAIL"
    else: 
        return "Inspection"
    

def save_image(img, filename):
    """이미지를 저장하는 함수"""
    if not os.path.exists("./capt"):
        os.makedirs("./capt")
    filepath = os.path.join("./capt", filename)
    cv2.imwrite(filepath, img)
    print(f"Image saved to {filepath}")

# 시리얼 포트 초기화
ser = serial.Serial("/dev/ttyACM0", 9600)

# API 엔드포인트 URL 및 인증 정보 설정
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/2912d85c-1351-4812-b14a-a5d1b590a56a/inference"
auth = HTTPBasicAuth("kdt2025_1-12", "CnLBBHS4v51kRIUFWZJn36qHibvg5ggc2Lwiu65L")

# 무한 루프 시작
while True:
    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        crop_info = {"x": 200, "y": 120, "width": 300, "height": 210}
        if crop_info is not None:
            img = crop_img(img, crop_info)
        result = inference_request(img)
        if result:
            df = display_results_table_in_new_window(result)  # 결과를 pandas DataFrame으로 출력
            draw_boxes(img, result)
            cv2.imshow("Annotated Image", cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3)))  # 크기를 3배로 증가
            save_image(img, "annotated_image.jpg")  # 주석 추가된 이미지를 저장

        # Enter 키가 정상 작동하도록 수정
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

cv2.destroyAllWindows()
ser.close()
