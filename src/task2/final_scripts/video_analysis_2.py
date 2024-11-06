import os

import cv2
import numpy as np
# from sticker_calibration import detect_stickers, draw_stickers
# from detect_corners3 import detect_chessboard_corners, refine_corners, get_warped_image, draw_chessboard, draw_refined_corners
# from camera_calibration import calibrate_camera, undistort_frame

from corners_detection import detect_corners, detect_chessboard_corners_extremities, refine_corners, draw_labeled_chessboard, draw_refined_corners
from stickers_detection import detect_stickers, draw_stickers
from chessboard_homography import compute_homography, compute_pose, draw_axis

from utils import resize_frame

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        print(f"Error code: {cap.get(cv2.CAP_PROP_FOURCC)}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize tracking variables
    last_frame_corners_extremities = None
    frame_count = 0
    
    # Cache to store the last valid detections
    cache = {
        'frame': None,

        'H': None,
        'objp': None,
        'rvec': None,
        'tvec': None,

        'chessboard_corners_extremities': None,
        'blue_stickers': None,
        'pink_stickers': None,
        'labeled_corners': None,
    }
    # last_frame = None

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        ###########################################
        # * DETECTION AND TRACKING
        ###########################################

        chessboard_corners = detect_corners(frame)
        
        chessboard_corners_extremities = None
        # Detect chessboard corners
        if chessboard_corners is not None:
            chessboard_corners_extremities = detect_chessboard_corners_extremities(frame, chessboard_corners)

        radius = 15  # Default search radius for corner refinement

        # If detection fails, use last known corners
        if chessboard_corners_extremities:
            cache['chessboard_corners_extremities'] = chessboard_corners_extremities

        elif chessboard_corners_extremities is None or any(corner is None for corner in chessboard_corners_extremities) and cache['chessboard_corners_extremities'] is not None:
            chessboard_corners_extremities = cache['chessboard_corners_extremities']
            radius = 40  # Increase search radius when using last known corners
        
        # Refine corner positions
        chessboard_corners_refined = refine_corners(frame, chessboard_corners_extremities, search_radius=radius)

        cache['chessboard_corners_extremities'] = chessboard_corners_refined


        ###########################################
        # * STICKER DETECTION AND CORNER LABELING
        ###########################################
        
        # Detect colored stickers and label corners
        blue_stickers, pink_stickers, labeled_corners = detect_stickers(frame, chessboard_corners_refined)
        
        # Update cache only with valid detections
        cache['blue_stickers'] = blue_stickers if blue_stickers else cache['blue_stickers']
        cache['pink_stickers'] = pink_stickers if pink_stickers else cache['pink_stickers']
        cache['labeled_corners'] = labeled_corners if all(corner is not None for corner in labeled_corners.values()) else cache['labeled_corners']


        ###########################################
        # * HOMOGRAPHY AND POSE ESTIMATION
        ###########################################
        
        # Compute homography matrix
        H = None
        objp = None

        if chessboard_corners is not None:
            H, objp = compute_homography(chessboard_corners)
        # print(f'Homography Matrix= {H}, {homography_corners}, {objp}')
        
        # Update cache with valid homography results
        if H is not None:
            cache['H'] = H
            cache['objp'] = objp

        # Compute and draw 3D pose
        if cache['labeled_corners'] is not None and cache['objp'] is not None:
            rvec, tvec = compute_pose(cache['objp'], cache['labeled_corners'])
            cache['rvec'] = rvec
            cache['tvec'] = tvec


        ###########################################
        # * VISUALIZATION
        ###########################################

        if cache['chessboard_corners_extremities'] is not None and chessboard_corners_refined is not None:
            frame = draw_refined_corners(frame, cache['chessboard_corners_extremities'], chessboard_corners_refined, search_radius=radius)

        if all(corner is not None for corner in cache['labeled_corners'].values()):
            frame = draw_labeled_chessboard(frame, cache['labeled_corners'])

        if cache['blue_stickers'] is not None and cache['pink_stickers'] is not None:
            frame = draw_stickers(frame, cache['blue_stickers'], cache['pink_stickers'])

        if cache['rvec'] is not None and cache['tvec'] is not None:
            frame = draw_axis(frame, cache['rvec'], cache['tvec'])

        
        ###########################################
        # * DISPLAY RESULTS
        ###########################################

        frame = resize_frame(frame, 1200)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


        ###########################################
        # * SAVE RESULTS
        ###########################################
        
        # Save every 100th frame
        os.makedirs('images_1', exist_ok=True)
        if frame_count % 100 == 0:
            if frame is not None:
                frame_name = f"frame_{frame_count:06d}.png"
                cv2.imwrite(os.path.join('images_1', frame_name), frame)
                print(f"Saved: {frame_name}")

        frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = ('C:/Users/VaryaStrizh/CV/elen0016-computer-vision-tutorial-master/elen0016-computer-vision-tutorial'
                  '-master/project/task2/videos/moving_2.mov')  # Remplacez par le chemin de votre vidéo
    

    video_path = 'videos/moving_game.mov'

    process_video(video_path)

# Note: Assurez-vous que les fonctions importées (detect_stickers, detect_chessboard, calibrate_camera)
# sont adaptées pour fonctionner sur une seule image à la fois.
