# Autonomous Car Project

<img src="https://github.com/muqeet-here/Nano-Vision-EV2/blob/main/Images/Cover.jpg" alt="Autonomous Car" style="width:820px; height:312px;">

## Overview

Welcome to the Autonomous Car Project! This project utilizes YOLOv5 for signal and speed sign detection, HSV for signal color detection, lane detection, and PyTesseract for speed sign limit acquisition. The goal is to create a self-driving car prototype with advanced computer vision capabilities.

## Features

- **YOLOv5 Detection:** The car employs YOLOv5 for accurate and real-time detection of traffic signals and speed signs, ensuring reliable navigation.

- **HSV Color Detection:** Signal color recognition and lane detection are achieved using the HSV color space, allowing the car to interpret and respond to different signal states and follow lanes accurately.

- **PyTesseract for Speed Sign Limits:** PyTesseract, an OCR (Optical Character Recognition) tool, is used to extract speed limits from speed signs, contributing to compliance with road regulations.

- **Lane Following:** The car utilizes HSV-based lane detection algorithms for precise and controlled lane following, enhancing navigation capabilities.

- **Pothole Detection:** Sensors are integrated to identify and navigate around potholes, providing a smoother ride.

- **Distance Maintenance:** VL53L0X sensors are utilized to maintain a safe distance from the vehicle ahead, enhancing traffic safety.

## Dependencies

To run this project, you'll need the following libraries:

- [OpenCV](https://github.com/opencv/opencv): A computer vision library for image and video processing.
- [NumPy](https://numpy.org/): A powerful library for numerical operations in Python.
- [YOLOv5](https://github.com/ultralytics/yolov5): YOLO (You Only Look Once) is a real-time object detection system, and version 5 is used in this project.
- [PyTorch](https://pytorch.org/): An open-source machine learning library used for YOLOv5 and other deep learning tasks.
- [torchvision](https://pytorch.org/vision/stable/index.html): A PyTorch package containing popular datasets, model architectures, and image transformations.
- [torchaudio](https://pytorch.org/audio/stable/index.html): An audio processing library built on PyTorch.

To install these dependencies, you can use the following commands:

```bash
# Install OpenCV and NumPy
pip install opencv-python
pip install numpy

# Install PyTorch, torchvision, and torchaudio
pip install torch torchvision torchaudio

# Install YOLOv5
pip install git+https://github.com/ultralytics/yolov5.git
