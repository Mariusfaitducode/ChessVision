import os

import cv2
import numpy as np
# from sticker_calibration import detect_stickers, draw_stickers
# from detect_corners3 import detect_chessboard_corners, refine_corners, get_warped_image, draw_chessboard, draw_refined_corners
# from camera_calibration import calibrate_camera, undistort_frame

from corners_detection import detect_chessboard_corners, refine_corners, get_warped_image, draw_labeled_chessboard, draw_refined_corners
from stickers_detection import detect_stickers, draw_stickers
from chessboard_homography import compute_homography, compute_pose, draw_axis

from utils import resize_frame

def process_video(video_path):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Vérifier si la vidéo est ouverte correctement
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
        print(f"Code d'erreur : {cap.get(cv2.CAP_PROP_FOURCC)}")
        return
    
    # Obtenir des propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # last_area = None # Used for check the area of the chessboard

    last_frame_corners = None
    # last_successful_frame = None
    frame_count = 0
    cache = {
        'frame': None,
        'H': None,
        'homography_corners': None,
        'objp': None,
        'chessboard_corners': None,
        'blue_stickers': None,
        'pink_stickers': None,
        'labeled_corners': None,

    }
    last_frame = None
    while True:

        # Lire la frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # * IMAGE PROCESSING
        
        # Appliquer la détection de l'échiquier
        chessboard_corners = detect_chessboard_corners(frame)
        radius = 15

        if chessboard_corners:
            # print(chessboard_corners)
            last_frame_corners = chessboard_corners
            last_frame = frame
            cache['frame'] = frame
            cache['chessboard_corners'] = chessboard_corners
        elif chessboard_corners is None or any(corner is None for corner in chessboard_corners):
            chessboard_corners = last_frame_corners
            frame = last_frame
            radius = 40
        
        chessboard_corners_refined = refine_corners(frame, chessboard_corners, search_radius=radius)

        # Appliquer la détection des autocollants
        blue_stickers, pink_stickers, labeled_corners = detect_stickers(frame, chessboard_corners_refined)
        # if blue_stickers:
        cache['blue_stickers'] = blue_stickers if blue_stickers else cache['blue_stickers']
        cache['pink_stickers'] = pink_stickers if pink_stickers else cache['pink_stickers']
        cache['labeled_corners'] = labeled_corners if all(corner is not None for corner in labeled_corners.values()) else cache['labeled_corners']
        # pink_stickersif pink_stickers:


        # Calculer l'homographie et la pose
        H, homography_corners, objp = compute_homography(frame)
        # print(f'Homography Matrix= {H}, {homography_corners}, {objp}')
        if frame_count == 1293:
            print()
        if H is not None:
            cache['H'] = H
            cache['homography_corners'] = homography_corners
            cache['objp'] = objp

        if all(corner is not None for corner in cache['labeled_corners'].values()) and cache['H'] is not None:
            print(f"Frame {frame_count}: {cache['labeled_corners']}")
            rvec, tvec = compute_pose(cache['objp'], cache['labeled_corners'])
            frame = draw_axis(frame, rvec, tvec)
            frame = draw_refined_corners(frame, cache['chessboard_corners'], chessboard_corners_refined, search_radius=radius)
            frame = draw_labeled_chessboard(frame, cache['labeled_corners'])
            frame = draw_stickers(frame, cache['blue_stickers'], cache['pink_stickers'])

            last_frame_corners = chessboard_corners_refined

            # * DISPLAY RESULTS
            frame = resize_frame(frame, 1200)
            # last_frame = frame.copy()

        else:
            print(f"Frame {frame_count}: Incomplete corners, skipping pose computation")
            frame_count += 1
            continue


        os.makedirs('images_1', exist_ok=True)
        # * SAVE IMAGE EVERY 100th FRAME
        if frame_count % 100 == 0:
            if frame is not None:
                frame_name = f"frame_{frame_count:06d}.png"
                cv2.imwrite(os.path.join('images_1', frame_name), frame)
                print(f"Saved: {frame_name}")

        # Afficher l'image traitée
        # cv2.imshow('Processed Video', frame)
        #
        # #Save the frames names
        #
        # # Attendre 1ms entre chaque frame et vérifier si l'utilisateur veut quitter
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q') or cv2.getWindowProperty('Processed Video', cv2.WND_PROP_VISIBLE) < 1:
        #     break

        frame_count += 1
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = ('C:/Users/VaryaStrizh/CV/elen0016-computer-vision-tutorial-master/elen0016-computer-vision-tutorial'
                  '-master/project/task2/videos/moving_2.mov')  # Remplacez par le chemin de votre vidéo
    process_video(video_path)

# Note: Assurez-vous que les fonctions importées (detect_stickers, detect_chessboard, calibrate_camera)
# sont adaptées pour fonctionner sur une seule image à la fois.
