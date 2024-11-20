from threading import Thread
from queue import Queue
import cv2


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
