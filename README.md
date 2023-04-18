# YOLOv8-browser

This repo shows that [YOLOv8](https://github.com/ultralytics/ultralytics) works in the browser with small engineering, when used in combination with [pyscript](https://pyscript.net/) and [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web).

![! ONNX YOLOv8 Object Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/raw/main/doc/img/detected_objects.jpg)
*Original image: [https://www.flickr.com/photos/nicolelee/19041780](https://www.flickr.com/photos/nicolelee/19041780)*



# Demo

Anyway, run it on your browser !!

https://lilacs2039.github.io/YOLOv8-browser/index.html


# Installation/Usage
```shell
git clone https://github.com/lilacs2039/YOLOv8-browser.git
cd YOLOv8-browser
python -m http.server 8000
```


# Model conversion to ONNX format

The original YOLO model is in `pt` format, so please convert it to `onnx` format by referring to the [original repository](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)



# Issue

- It did not work with the **webgl backend** of onnxruntime-web. (Because certain operations of yolov8n were not supported.)



# Notes: Image Size

- The input images are resized to 640x480 px during inference.
- So that it might affect the accuracy of the model if the input image has a different aspect ratio compared.



# Credits

- Original repogitry : [ibaiGorordo](https://github.com/ibaiGorordo)/**[ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)**
- YOLOv8 model（[License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) : GPL-3.0）: https://github.com/ultralytics/ultralytics



