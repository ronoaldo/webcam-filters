import cv2
import virtualvideo
import mediapipe as mp
import numpy as np
import math
import sys
from typing import List, Mapping, Optional, Tuple, Union
from pprint import pprint
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

overlays = {
    "ILoveYou": cv2.imread("image/ILoveYou.png", flags=cv2.IMREAD_UNCHANGED)
}

INPUT_DEVICE=0
OUTPUT_DEVICE=10

# Borrowed from: 
#   https://github.com/google/mediapipe/blob/91589b10d3c684af00cf8e3d14e4683797ab55bd/mediapipe/python/solutions/drawing_utils.py#L49
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and \
            (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# God Bless Stack Overflow
def overlay(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def proccess(frame):
    frame_h, frame_w, _ = frame.shape
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = recognizer.recognize(image)
    if len(result.gestures) > 0:
        gesture = result.gestures[0][0].category_name
        if gesture in overlays:
            lm = result.hand_landmarks[0]
            x, y = _normalized_to_pixel_coordinates(lm[8].x, lm[8].y, frame_w, frame_h)
            x = max(x-150, 0)
            overlay(frame, overlays[gesture], x, y)
    return frame

def init_webcam():
    camera = cv2.VideoCapture(INPUT_DEVICE)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def debug():
    camera = init_webcam()
    while True:
        _, frame = camera.read()
        if frame is None:
            print("Error rendering frame: frame=None. Video not available?")
            break
    
        frame = proccess(frame)
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == 13:
            break

class FancyCam(virtualvideo.VideoSource):
    def __init__(self, device=INPUT_DEVICE):
        self.camera = init_webcam()
        _, frame = self.camera.read()
        if frame is None:
            raise("Error reading initial frame from wecbam. Perhaps video device is not ready?")
        size = frame.shape
        #opencv's shape is y,x,channels
        self._size = (size[1],size[0])

    def img_size(self):
        return self._size

    def fps(self):
        return 20

    def generator(self):
        while True:
            _, frame = self.camera.read()
            if frame is None:
                print("Error rendering frame: frame=None. Video not available?")
                break
            yield proccess(frame)

def virtual_webcam():
    vidsrc = FancyCam(device=INPUT_DEVICE)
    w, h = vidsrc.img_size()
    fvd = virtualvideo.FakeVideoDevice()
    fvd.init_input(vidsrc)
    fvd.init_output(OUTPUT_DEVICE, w, h, fps=30)
    try:
        fvd.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if "debug" in sys.argv:
        debug()
    else:
        virtual_webcam()
