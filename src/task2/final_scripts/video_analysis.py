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
    
    while True:

        # Lire la frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # * IMAGE PROCESSING
        
        # Appliquer la détection de l'échiquier
        chessboard_corners = detect_chessboard_corners(frame)
        radius = 15;

        if chessboard_corners is None and last_frame_corners is not None:
            # chessboard_corners_refined = refine_corners(frame, chessboard_corners, 15)
            chessboard_corners = last_frame_corners
            radius = 40

        
        chessboard_corners_refined = refine_corners(frame, chessboard_corners, search_radius=radius)



        # Appliquer la détection des autocollants
        blue_stickers, pink_stickers, labeled_corners = detect_stickers(frame, chessboard_corners_refined)


        # Calculer l'homographie et la pose
        H, homography_corners, objp = compute_homography(frame)

        if H is not None:
            print(labeled_corners)
            rvec, tvec = compute_pose(objp, labeled_corners)
        
        # * DRAW RESULTS

        # print(labeled_corners)

        frame = draw_axis(frame, rvec, tvec)


        frame = draw_stickers(frame, blue_stickers, pink_stickers)


        frame = draw_refined_corners(frame, chessboard_corners, chessboard_corners_refined, search_radius=radius)

        frame = draw_labeled_chessboard(frame, labeled_corners)

        last_frame_corners = chessboard_corners_refined



        warped_image = get_warped_image(frame, chessboard_corners_refined)



        # frame = draw_stickers(frame, blue_stickers, pink_stickers)
    
        # Dessiner l'échiquier s'il est détecté
        # if chessboard_corners is not None:
        #     frame = draw_chessboard(frame, chessboard_corners)


        # * DISPLAY RESULTS
        frame = resize_frame(frame, 1200)

        # Afficher l'image traitée
        cv2.imshow('Processed Video', frame)
        
        # Attendre 1ms entre chaque frame et vérifier si l'utilisateur veut quitter
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('Processed Video', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    process_video(video_path)

# Note: Assurez-vous que les fonctions importées (detect_stickers, detect_chessboard, calibrate_camera)
# sont adaptées pour fonctionner sur une seule image à la fois.
