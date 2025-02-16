import os
from datetime import datetime
import serial
import requests
import numpy
from io import BytesIO
from pprint import pprint

import cv2

ser = serial.Serial("/dev/ttyACM0", 9600)

# API endpoint
api_url = ""

def save_img(img, folder="captured_images"):
    """Save Image to Disk
    Args:
        img (numpy.array): Image numpy array
        folder (str): Folder to save images
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"image_test_{timestamp}.jpg")
    # Save the image
    cv2.imwrite(file_path, img)
    print(f"Image saved at: {file_path}")
    return file_path

def get_img():
    """Get Image From USB Camera

    Returns:
        numpy.array: Image numpy array
    """

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    return img


def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img


def inference_reqeust(img: numpy.array, api_rul: str):
    """_summary_

    Args:
        img (numpy.array): Image numpy array
        api_rul (str): API URL. Inference Endpoint
    """
    _, img_encoded = cv2.imencode(".jpg", img)

    # Prepare the image for sending
    img_bytes = BytesIO(img_encoded.tobytes())

    # Send the image to the API
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    print(files)

    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            pprint(response.json())
            return response.json()
            print("Image sent successfully")
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")


while 1:
    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        # crop_info = None
        crop_info = {"x": 200, "y": 120, "width": 300, "height": 210}

        if crop_info is not None:
            img = crop_img(img, crop_info)

         # Save the image before sending to API
        saved_path = save_img(img)

        cv2.imshow("", img)
        # this key
        cv2.waitKey(1)
        result = inference_reqeust(img, api_url)
        ser.write(b"1")
    else:
        pass
