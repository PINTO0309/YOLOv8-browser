import time
import cv2
import numpy as np

from js import js_init_session, js_run_session
from pyodide.ffi import create_proxy, to_js
import asyncio

from yolov8.utils import xywh2xyxy, nms, draw_detections
from yolov8.YOLOv8_base import YOLOv8Base

class YOLOv8(YOLOv8Base):
    async def init_session(self, path):
        self.session = await js_init_session(path)
        
    async def run_ssession(self, input_tensor):
        outputs_jsproxy = await js_run_session(to_js(np.array(input_tensor)))
        # JsProxy {'output0':ndarray object} -> [ndarray]
        output_info = {k:np.array(v) for k,v in outputs_jsproxy.to_py().items()}
        outputs = output_info['output0'].reshape(output_info['dims'])
        return outputs
    
    def get_input_details(self):
        self.input_names = self.session.inputNames

        self.input_shape = (1,3,480, 640)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        self.output_names = self.session.outputNames

