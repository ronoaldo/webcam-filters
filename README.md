# Webcam filter with hand gestures

## Install

Required tools:

1. Python 3 (3.12 not supported)
2. FFmpeg
3. Video4Linux Loopback module

## Usage

1. Download the models and save them into models/ directory:
    1. [Gesture](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task)
    2. [Face](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite)

2. Create a video4linux lookpack device:
    `sudo modprobe v4l2loopback video_nr=10 exclusive_caps=1 device_label="Webcam Filters"`

3. Close any apps currently using the camera

4. Launch the script with:
    `python webcam.py`
