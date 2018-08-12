"""
velocity_detector.py
"""

import cv2 as cv
import numpy as np

WIDTH = 160         # With of input image
HEIGHT = 120        # Height of input image

CAM = cv.VideoCapture("dkcar.mp4")
UBASE_WIDTH = 60    # Upper-base width
LBASE_WIDTH = 320   # Lower-base width
UOFFSET = 45        # Upper-base margin
LOFFSET = 20        # Lower-base margin
MAX_ACC = 0.2       # Max possible acceleration

SRC_UL = [(WIDTH - UBASE_WIDTH) / 2, UOFFSET]
SRC_LL = [(WIDTH - LBASE_WIDTH) / 2, HEIGHT - LOFFSET]
SRC_UR = [(WIDTH + UBASE_WIDTH) / 2, UOFFSET]
SRC_LR = [(WIDTH + LBASE_WIDTH) / 2, HEIGHT - LOFFSET]

DST_UL = [0, 0]
DST_LL = [0, HEIGHT]
DST_UR = [WIDTH, 0]
DST_LR = [WIDTH, HEIGHT]

DISPLACEMENT_CUTOFF = 0.2

def make_speed_detector():
    """Speed detector factory."""

    pts1 = np.float32([SRC_UL, SRC_LL, SRC_UR, SRC_LR])
    pts2 = np.float32([DST_UL, DST_LL, DST_UR, DST_LR])
    M = cv.getPerspectiveTransform(pts1, pts2)

    prev = None
    v_last = 0.0

    def detect(image):
        """Detect speed from images"""
        nonlocal prev, v_last

        if prev is None:
            prev = curr
            v_last = 0.0
            return v_last

        flow = cv.calcOpticalFlowFarneback(
            prev,   # Previous image
            curr,   # Current image
            None,   # Computed flow image that has the same size oas prev and type CV_32FC2.
            0.5,    # Specifies the image scale (<1) to build pyramids for each image.
            3,      # Number of pyramid layers including the initial image.
            15,     # winsize, averaging windows size.
            3,      # iterations, number of iterations the algorithm does at each pyramid level.
            5,      # standard deviation of the Gaussian that is used to smooth derivative
            1.2,
            0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        v = mag * np.sin(ang)

        v = v[np.where(v >= DISPLACEMENT_CUTOFF)]
        v_max = v_last + MAX_ACC*2
        v_min = max(v_last - MAX_ACC*2, DISPLACEMENT_CUTOFF)
        v = np.clip(v, v_min, v_max)
        if v.size > 0:
            v_avg = v.mean()
        else:
            v_avg = max(v_last - MAX_ACC, 0)

        prev = curr
        v_last = v_avg
        return v_last

    return detect

get_speed = make_speed_detector()

