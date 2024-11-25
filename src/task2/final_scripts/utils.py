from threading import Thread
from queue import Queue

import cv2
import numpy as np


def find_corners_with_timeout(gray, chessboard_size, flags):
    result_queue = Queue()
    
    def worker():
        result = cv2.findChessboardCorners(gray, chessboard_size, flags)
        result_queue.put(result)
    
    thread = Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait max 0.05 seconds
    thread.join(timeout=0.05)
    
    if result_queue.empty():
        return False, None
    
    return result_queue.get()


def resize_frame(frame, display_width):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    display_height = int(display_width / aspect_ratio)
    return cv2.resize(frame, (display_width, display_height))



def extrapolate_point(p1, p2):
    """
    Extrapole un point en utilisant la direction définie par deux points.
    """
    return p1 + (p1 - p2)


def extrapolate_point_with_ratio(p1, p2, p3):

    v1 = p1 - p2
    v2 = p3 - p2

    ratio = np.linalg.norm(v1) / np.linalg.norm(v2)

    return p1 + v1 * ratio