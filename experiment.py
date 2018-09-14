#!/usr/bin/env python3
"""
Experimental routines

Usage:
    experimental.py (show-lane) [--img=<image_path>] [--pause]
    experimental.py (show) [--tub=<tub1,tub2,..tubn>] [--pause]

Options:
    -h --help        Show this screen.
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from donkeychainer import dataset as ds
from donkeychainer import cvtool as cvt

def show_and_wait(title, image):
    cv2.imshow('Lane Detection', image)
    cv2.waitKey(0)

def show_lane(image_path, pause=False):
    # Prepare mask of shape (120, 160)
    mask = cv2.imread('./image_mask.jpg')
    mask = cvt.discard_colors(mask)
    print('shape of mask:', mask.shape)
    mask = mask.astype(float)
    mask *= 1/255.

    # Load an color image in grayscale
    image = cv2.imread(image_path)

    lines = detect_lines(image, mask, pause)
    draw_lines(image, lines, pause)

def show(tub_names):
    # Prepare mask of shape (120, 160)
    mask = cv2.imread('./image_mask.jpg')
    mask = cvt.discard_colors(mask)
    print('shape of mask:', mask.shape)
    mask = mask.astype(float)
    mask *= 1/255.

    dataset = ds.load_data(tub_names)
    for data in dataset:
        image = data[0][0];
        image = image.transpose((1, 2, 0))
        image = np.asarray(image * 255, dtype=np.uint8)

        lines = detect_lines(image, mask, pause=False)
        draw_lines(image, lines, pause=False)

        cv2.imshow("yay", image)
        cv2.waitKey(100)

def draw_lines(image, lines, pause):
    for line in lines:
        cvt.draw_line(image, line)

    if pause:
        show_and_wait('image with lines', image)

    return image

def detect_lines(image, mask, pause=False):
    print('shape:', image.shape)
    if pause:
        show_and_wait('original', image)

    image = cvt.discard_colors(image)
    if pause:
        show_and_wait('grey',image)

    image = cvt.remove_noise(image, 5)
    if pause:
        show_and_wait('remove noise', image)

    image = cvt.detect_edges(image, low_threshold=50, high_threshold=150)
    if pause:
        show_and_wait('detect edges',image)

    image = np.asarray((image * mask), dtype=np.uint8)
    if pause:
        show_and_wait('mask', image)

    rho = 0.8
    theta = np.pi/180
    threshold = 15
    min_line_len = 20
    max_line_gap = 100
    lines = cvt.hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
    cvt.draw_lines(image, lines)
    if pause:
        show_and_wait('hough lines',image)

    left_lines, right_lines = cvt.separate_lines(lines)
    print('left_lines:', left_lines)
    print('right_lines:', right_lines)
    left = []
    right = []
    if len(right_lines) > 0:
        right = cvt.reject_outliers(right_lines,  cutoff=(0.20, 0.90))
    if len(left_lines) > 0:
        left = cvt.reject_outliers(left_lines, cutoff=(-0.90, -0.20))

    lines = []
    if len(left) >= 4:
        lines.append(left)
    if len(right) >= 4:
        lines.append(right)
    return lines

if __name__ == '__main__':
    args = docopt(__doc__)

    if args['show-lane']:
        image_path = args['--img']
        pause = args['--pause']
        show_lane(image_path, pause)
    elif args['show']:
        tub = args['--tub']
        show(tub)

    cv2.destroyAllWindows()
