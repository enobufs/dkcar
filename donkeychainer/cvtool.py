import cv2
import numpy as np

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_line(image, line, color=[255, 0, 255], thickness=2):
    print("input image shape:", image.shape)
    print("input image type :", image.dtype)
    print("input image ndim :", image.ndim)
    x1, y1, x2, y2 = int(line[0]),int(line[1]),int(line[2]),int(line[3])
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def slope(x1, y1, x2, y2):
    try:
        return (y1 - y2) / (x1 - x2)
    except:
        return 0
        

def separate_lines(lines):
    right = []
    left = []

    if lines is not None:
        for x1,y1,x2,y2 in lines[:, 0]:
            m = slope(x1,y1,x2,y2)
            if m >= 0:
                right.append([x1,y1,x2,y2,m])
            else:
                left.append([x1,y1,x2,y2,m])
    return left, right


def reject_outliers(data, cutoff, threshold=0.08, lane='left'):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    try:
        if lane == 'left':
            return data[np.argmin(data,axis=0)[-1]]
        elif lane == 'right':
            return data[np.argmax(data,axis=0)[-1]]
    except:
        return []


def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y

def make_velocity_detector(debug=False):
    """Velocity detector factory."""

    WIDTH = 160         # With of input image
    HEIGHT = 120        # Height of input image

    INPUT_PATH="dkcar.mp4"

    UBASE_WIDTH = 120   # Upper-base width
    LBASE_WIDTH = 320   # Lower-base width
    UOFFSET = 50        # Upper-base margin
    LOFFSET = 20        # Lower-base margin
    MAX_ACC = 0.4       # Max possible acceleration

    SRC_UL = [(WIDTH - UBASE_WIDTH) / 2, UOFFSET]
    SRC_LL = [(WIDTH - LBASE_WIDTH) / 2, HEIGHT - LOFFSET]
    SRC_UR = [(WIDTH + UBASE_WIDTH) / 2, UOFFSET]
    SRC_LR = [(WIDTH + LBASE_WIDTH) / 2, HEIGHT - LOFFSET]

    DST_UL = [0, 0]
    DST_LL = [0, HEIGHT]
    DST_UR = [WIDTH, 0]
    DST_LR = [WIDTH, HEIGHT]

    VELOCITY_CUTOFF_PCT = 67

    pts1 = np.float32([SRC_UL, SRC_LL, SRC_UR, SRC_LR])
    pts2 = np.float32([DST_UL, DST_LL, DST_UR, DST_LR])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    prev = None
    v_last = 0.0

    def get_velocity(image):
        """Detect velocity from images"""
        nonlocal prev, v_last, debug
        hsv_bgr = None
        top_view = None

        curr = cv2.warpPerspective(image, M, (160, 120))

        if debug:
            top_view = np.copy(curr)

        """
	# find the colors within the specified boundaries and apply
	# the mask
        y_l = np.array([48, 121, 133], dtype = "uint8")
        y_h = np.array([89, 224, 247], dtype = "uint8")
        mask = cv2.inRange(top_view, y_l, y_h)
        curr = cv2.bitwise_and(top_view, top_view, mask = mask)
        """

        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        #show_and_wait(curr)
        #curr = cv2.GaussianBlur(curr, (5, 5), 0)
        #show_and_wait(curr)
        #curr = cv2.Canny(curr, 50, 150)
        #show_and_wait(curr)

        if prev is None:
            prev = curr
            v_last = 0.0
            return v_last, top_view, np.zeros_like(image)

        flow = cv2.calcOpticalFlowFarneback(
            prev,   # Previous image
            curr,   # Current image
            None,   # Computed flow image that has the same size oas prev and type CV_32FC2.
            0.5,    # Specifies the image scale (<1) to build pyramids for each image.
            3,      # Number of pyramid layers including the initial image.
            15,     # winsize, averaging windows size.
            3,      # iterations, number of iterations the algorithm does at each pyramid level.
            5,      # standard deviation of the Gaussian that is used to smooth derivative
            1.5,
            0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag[mag < 0.2] = 0
        v = mag * np.sin(ang)


        if debug:
            ######################
            ## Histogram for mag
            ar = np.arange(-20.0, 20.0, 0.50, dtype=np.float)
            his = np.histogram(v, bins=ar)

            for i, n in enumerate(his[0]):
                bgr = (255, 255, 0)
                if his[1][i] < 0:
                    bgr = (0, 255, 255)

                #print('[{}] {} - {}'.format(i, n, his[1][i]))
                cv2.rectangle(   image, #top_view,
                                (i*2, HEIGHT),
                                (i*2, HEIGHT - int(n / 10)),
                                bgr, #(0, 255, 255),
                                cv2.FILLED)

            hsv = np.zeros_like(image)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(np.abs(v), None, 0, 255, cv2.NORM_MINMAX)
            hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ##
            ######################

        v_abs = np.absolute(v)
        v = v[v_abs >= np.percentile(v_abs, VELOCITY_CUTOFF_PCT)]

        v_max = v_last + MAX_ACC
        v_min = v_last - MAX_ACC
        v = np.clip(v, v_min, v_max)
        if v.size > 0:
            v_avg = v.mean()
        else:
            if v_last > 0:
                v_avg = max(v_last - MAX_ACC, 0)
            elif v_last < 0:
                v_avg = min(v_last + MAX_ACC, 0)
            else:
                v_avg = 0

        prev = curr
        v_last = v_avg
        return v_last, top_view, hsv_bgr

    return get_velocity

