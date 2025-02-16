import time
import serial
import requests
import numpy as np
from io import BytesIO
from pprint import pprint
import matplotlib.pyplot as plt
import random
from requests.auth import HTTPBasicAuth
import pandas as pd
import os

def display_results_table_in_new_window(results):
    """결과를 별도의 창에 표 형태로 출력하는 함수"""
    if results is None or 'objects' not in results or not results['objects']:
        print("No valid objects to display.")
        return

    data = []
    for obj in results['objects']:
        if obj is not None and 'class' in obj and 'score' in obj and 'box' in obj:
            data.append({
                'Class': obj['class'],
                'Score': f"{obj['score']:.2f}",
                'Box': obj['box']
            })

    df = pd.DataFrame(data)

    # 별도의 창에서 출력
    print("\n=== Detection Results ===")
    print(df)
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.show()

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

def inference_request(img: np.array):
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

def generate_random_color():
    """랜덤 색상 생성"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_boxes(img, results):
    """API 응답에 따라 이미지에 경계 상자와 클래스 이름을 표시하는 함수"""
    if results is None or 'objects' not in results or not results['objects']:
        print("No valid objects found in results.")
        return

    for obj in results['objects']:
        if obj is None or 'class' not in obj or 'box' not in obj:
            continue
        box = obj['box']
        class_name = obj['class'][:3]
        score = obj['score']
        color = generate_random_color()
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_image(img, filename):
    """이미지를 저장하는 함수"""
    if not os.path.exists("./capt"):
        os.makedirs("./capt")
    filepath = os.path.join("./capt", filename)
    cv2.imwrite(filepath, img)
    print(f"Image saved to {filepath}")

def show_image_with_plt(img):
    """이미지를 matplotlib으로 표시하는 함수"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# 시리얼 포트 초기화
ser = serial.Serial("/dev/ttyACM0", 9600)

# API 엔드포인트 URL 및 인증 정보 설정
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/2912d85c-1351-4812-b14a-a5d1b590a56a/inference"
auth = HTTPBasicAuth("kdt2025_1-12", "zsX52Atrhe29cFY3B6bvL2LkufSMySDi110r3yQx")

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
            draw_boxes(img, result)
            display_results_table_in_new_window(result)  # 결과를 새 창에 표 형태로 출력
            save_image(img, "annotated_image.jpg")  # 주석 추가된 이미지를 저장
            show_image_with_plt(img)  # 이미지를 matplotlib으로 표시
        else:
            print("No valid result from API.")

        # Enter 키로 컨베이어 벨트 작동
        input("Press Enter to start conveyor belt...")
        ser.write(b"1")

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cv2.destroyAllWindows()
ser.close()
