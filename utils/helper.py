
import cv2
import numpy as np


def draw_boxes(image_copy, bounds, color=(0, 255, 0), thickness=5):
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        pts = np.array([p0, p1, p2, p3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        image_copy = cv2.polylines(image_copy, [pts], isClosed=True, color=color, thickness=thickness)

    return image_copy
