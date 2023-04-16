import time
import cv2
import numpy as np
import onnxruntime

from yolov8.utils import xywh2xyxy, nms, draw_detections
from yolov8.YOLOv8_base import YOLOv8Base

class YOLOv8(YOLOv8Base):
    async def init_session(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        
    async def run_ssession(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/yolov8m.onnx"

    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    # Detect Objects
    yolov7_detector(img)

    # Draw detections
    combined_img = yolov7_detector.draw_detections(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
