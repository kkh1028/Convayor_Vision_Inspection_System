import time
import serial
import requests
import numpy as np
from io import BytesIO
import cv2
from requests.auth import HTTPBasicAuth

# 시리얼 포트 초기화
ser = serial.Serial("/dev/ttyACM0", 9600)
# API 엔드포인트 및 인증 정보
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/2912d85c-1351-4812-b14a-a5d1b590a56a/inference"
auth = HTTPBasicAuth("kdt2025_1-12", "CnLBBHS4v51kRIUFWZJn36qHibvg5ggc2Lwiu65L")

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
    return img[y:y + h, x:x + w]

def inference_request(img: np.array):
    """이미지를 API에 전송하여 추론 요청"""
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = BytesIO(img_encoded.tobytes())
    headers = {
        'Content-Type': 'image/jpeg',
    }
    try:
        response = requests.post(api_url, auth=auth, headers=headers, data=img_bytes.getvalue())
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None
    
def draw_boxes(img, results):
    """API 응답에 따라 이미지에 경계 상자와 클래스 이름을 표시"""
    if "objects" in results:
        for obj in results["objects"]:
            box = obj["box"]
            class_name = obj["class"][:4]
            score = obj["score"]
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        print("No objects found in the API response.")
# 메인 루프
while True:
    data = ser.readline().strip()
    print(f"Received: {data}")
    if data == b"0":
        img = get_img()
        crop_info = {"x": 200, "y": 100, "width": 300, "height": 300}
        if crop_info is not None:
            img = crop_img(img, crop_info)
        cv2.imshow("Cropped Image", img)
        cv2.waitKey(1)
        result = inference_request(img)
        if result:
            draw_boxes(img, result)
            cv2.imshow("Annotated Image", img)
            cv2.waitKey(1)
        ser.write(b"1")
    else:
        time.sleep(0.1)