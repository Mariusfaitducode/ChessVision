import os

import cv2
import numpy as np
# from sticker_calibration import detect_stickers, draw_stickers
# from detect_corners3 import detect_chessboard_corners, refine_corners, get_warped_image, draw_chessboard, draw_refined_corners
# from camera_calibration import calibrate_camera, undistort_frame

from corners_detection import *
from corners_tracking2 import *

from stickers_detection import detect_stickers, draw_stickers, label_corners
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
    frame_interval = 25
    
    # Cache to store the last valid detections
    cache = {
        'frame': None,

        'last_frame':None,
        'extended_grid': None,
        'extended_mask': None,

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

    skip_moment = False

    while True:

        frame_count += 1

        if frame_count % frame_interval != 0:
            continue


        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()

        if not ret:
            break

        ###########################################
        # * STICKERS DETECTION
        ###########################################

        # Detect colored stickers and label corners
        blue_stickers, pink_stickers = detect_stickers(frame)
        

        ###########################################
        # * CORNERS DETECTION AND TRACKING
        ###########################################

        chessboard_corners = detect_corners(frame)
        
        chessboard_corners_extremities = None

        

        # Detect chessboard corners
        if chessboard_corners is not None: # * Found corners with cv2.findChessboardCorners

            # chessboard_corners_extremities = detect_chessboard_corners_extremities(frame, chessboard_corners)
            extended_grid, chessboard_corners_extremities = detect_all_chessboard_corners(frame, chessboard_corners)

            # img = draw_all_corners(frame, extended_grid)
            # img = draw_extremities(frame, chessboard_corners_extremities)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)


            if chessboard_corners_extremities: # * SAVE DETECTION INFORMATIONS
                cache['chessboard_corners_extremities'] = chessboard_corners_extremities
                cache['extended_mask'] = np.ones((9, 9), dtype=bool)
                cache['extended_grid'] = extended_grid
                cache['last_frame'] = frame


        else: 
            print("NO CORNERS FOUND")

            # Skip option

            chessboard_corners_extremities = cache['chessboard_corners_extremities']
            # skip_moment = True
            # continue

            # * SEARCH CORNERS USING LAST KNOWN CORNERS AND CORNERS TRACKING

            # chessboard_corners_extremities = None

            if cache['extended_grid'] is not None and cache['last_frame'] is not None:

                # if cache['extended_mask'] is None:
                    
                
                extremities, grid, mask = estimate_corners_movement(cache['extended_grid'], cache['extended_mask'], frame, cache['last_frame'], debug=False)

                # extremities = None
                # print('corners', estimated_corners)

                cache['extended_grid'] = grid
                cache['extended_mask'] = mask
                cache['last_frame'] = frame

                if extremities is not None:

                    # img = draw_all_corners(frame, extremities)

                    chessboard_corners_extremities = extremities
                    cache['chessboard_corners_extremities'] = chessboard_corners_extremities
                    

            
        # Refine corner positions
        # chessboard_corners_refined = refine_corners(frame, chessboard_corners_extremities, search_radius=radius)

        chessboard_corners_refined = []

        if chessboard_corners_extremities is not None:

            for corner in chessboard_corners_extremities:
                chessboard_corners_refined.append(tuple(corner))

        cache['chessboard_corners_extremities'] = chessboard_corners_refined


        ###########################################
        # * CORNER LABELING
        ###########################################
        
        # TODO : label corners without using stickers

        # labeled_corners = None
        labeled_corners = label_corners(chessboard_corners_refined, blue_stickers, pink_stickers)
        # labeled_corners = label_corners2(chessboard_corners_refined)

        # Update cache only with valid detections
        cache['blue_stickers'] = blue_stickers if blue_stickers else cache['blue_stickers']
        cache['pink_stickers'] = pink_stickers if pink_stickers else cache['pink_stickers']

        if labeled_corners is not None:
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

        warped_frame = None

        if cache['labeled_corners'] is not None and all(corner is not None for corner in cache['labeled_corners'].values()):
            warped_frame = get_warped_image(frame, cache['labeled_corners'])
            frame = draw_labeled_chessboard(frame, cache['labeled_corners'])

        if cache['chessboard_corners_extremities'] is not None and chessboard_corners_refined is not None:
            # frame = draw_refined_corners(frame, cache['chessboard_corners_extremities'], chessboard_corners_refined, search_radius=radius)
            frame = draw_corners(frame, chessboard_corners_refined)

        if cache['blue_stickers'] is not None and cache['pink_stickers'] is not None:
            frame = draw_stickers(frame, cache['blue_stickers'], cache['pink_stickers'])

        if cache['rvec'] is not None and cache['tvec'] is not None:
            frame = draw_axis(frame, cache['rvec'], cache['tvec'])

        
        ###########################################
        # * DISPLAY RESULTS
        ###########################################

        frame = resize_frame(frame, 1200)
        cv2.imshow('frame', frame)

        cv2.imshow('warped_frame', warped_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


        ###########################################
        # * SAVE RESULTS
        ###########################################
        
        # Save every 100th frame
        os.makedirs('images_1', exist_ok=True)
        if frame_count % 50 == 0 or skip_moment:
            if frame is not None:
                frame_name = f"frame_{frame_count:06d}.png"
                cv2.imwrite(os.path.join('images_results/frames', frame_name), frame)
                print(f"Saved: {frame_name}")

            if warped_frame is not None:
                warped_frame_name = f"warped_frame_{frame_count:06d}.png"
                cv2.imwrite(os.path.join('images_results/warped_images', warped_frame_name), warped_frame)
                print(f"Saved: {warped_frame_name}")

            skip_moment = False

        # frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = ('C:/Users/VaryaStrizh/CV/elen0016-computer-vision-tutorial-master/elen0016-computer-vision-tutorial'
                  '-master/project/task2/videos/moving_2.mov')  # Remplacez par le chemin de votre vidéo
    

    video_path = 'videos/moving_game.mov'

    process_video(video_path)
