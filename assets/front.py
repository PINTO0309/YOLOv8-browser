import asyncio

yolov7_detector = None
img = None

async def main():
    # setup ----------------------------------------------------------------
    print('importing python packages...')
    import pyodide_js
    await pyodide_js.loadPackage(['opencv-python', "matplotlib"])

    from pyodide.ffi import create_proxy
    from assets.YOLOv8_browser import YOLOv8
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from js import document, readFileAsDataURL
    import base64
    import traceback

    # functions ----------------------------------------------------------------
    async def detect_from_upload(e):
        print('detect_from_upload...')
        try:
            dataurl = await readFileAsDataURL()
            await detect_from_dataURL(dataurl)
        except e:
            traceback.print_stack()
            raise e

    async def detect_from_dataURL(dataURL: str):
        binary_data = base64.b64decode(dataURL.split(",")[1])
        img_data = np.frombuffer(binary_data, np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        await detect(img)

    async def detect_from_imgurl(img_url: str):
        global img
        print(f'image url : {img_url}')
        img = cv2.imread(img_url)
        print('loaded image')
        await detect(img)

    async def detect(img: np.ndarray):
        print('Detecting Objects...')
        await yolov7_detector.detect_objects(img)

        print('Drawing detections...')
        combined_img = yolov7_detector.draw_detections(img)

        print('Ploting image...')
        _, buffer = cv2.imencode('.jpg', combined_img)
        jpg_as_text = base64.b64encode(buffer).decode()
        img_str = f'data:image/png;base64,{jpg_as_text}'
        img_tag = document.getElementById('image')
        img_tag.src = img_str
        print('Detection completed!')

    document.getElementById("fileInput").addEventListener(
        "change", create_proxy(detect_from_upload))
    # app ----------------------------------------------------------------
    print('running app...')
    model_path = "./models/yolov8n.onnx"
    # Initialize YOLOv7 object detector
    global yolov7_detector
    yolov7_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)
    await yolov7_detector.init()

    print('test run detection...')
    await detect_from_imgurl("assets/19041780_d6fd803de0_3k.jpg")

print('loaded front.py')
