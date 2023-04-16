from abc import ABC, abstractmethod
import time
import cv2
import numpy as np

from yolov8.utils import xywh2xyxy, nms, draw_detections


class YOLOv8Base(ABC):
    "abstract class for YOLOv8."
    # ----------------------- public methods -----------------------
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.path = path
    
    async def init(self):
        await self.initialize_model(self.path)

    async def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        # Perform inference on the image
        outputs = await self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids
    
    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    # ----------------------- abstract methods -----------------------
    @abstractmethod
    async def init_session(self, path):
        raise NotImplementedError()

    @abstractmethod
    async def run_ssession(self, input_tensor)->dict:
        raise NotImplementedError()

    @abstractmethod
    def get_input_details(self):
        "initialize self.input_names, self.input_shape, self.input_height, self.input_width"
        raise NotImplementedError()

    @abstractmethod
    def get_output_details(self):
        "initialize self.output_names"
        raise NotImplementedError()
    
    # ----------------------- private methods -----------------------
    async def initialize_model(self, path):
        await self.init_session(path)

        # Get model info
        self.get_input_details()
        self.get_output_details()


    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    async def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = await self.run_ssession(input_tensor)

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


