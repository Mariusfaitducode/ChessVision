import cv2
import numpy as np
from sticker_calibration import detect_stickers, draw_stickers
from detect_corners3 import detect_chessboard_corners, refine_corners, get_warped_image, draw_chessboard, draw_refined_corners
# from camera_calibration import calibrate_camera, undistort_frame



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
        
        # * Image processing
        # blue_stickers, pink_stickers, chessboard_corners, approx = process_frame(frame)

        # Appliquer la détection des autocollants
        blue_stickers, pink_stickers = detect_stickers(frame)
        
        # Appliquer la détection de l'échiquier
        chessboard_corners = detect_chessboard_corners(frame)

        if chessboard_corners is not None:
            chessboard_corners_refined = refine_corners(frame, chessboard_corners, 15)

            frame = draw_refined_corners(frame, chessboard_corners, chessboard_corners_refined, 15)
            last_frame_corners = chessboard_corners_refined

        elif last_frame_corners is not None:
            # chessboard_corners_refined = refine_corners(frame, last_frame_corners, 40)

            # frame = draw_refined_corners(frame, last_frame_corners, chessboard_corners_refined, 40)
            # last_frame_corners = chessboard_corners_refined

            chessboard_corners_refined = last_frame_corners

        warped_image = get_warped_image(frame, chessboard_corners_refined)
        
        # * Draw results

        # frame = draw_stickers(frame, blue_stickers, pink_stickers)
    
        # Dessiner l'échiquier s'il est détecté
        # if chessboard_corners is not None:
        #     frame = draw_chessboard(frame, chessboard_corners)


        # * Display
        
        # Redimensionner l'image pour l'affichage
        display_width = 1200  # Vous pouvez ajuster cette valeur selon vos besoins
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_height = int(display_width / aspect_ratio)
        display_frame = cv2.resize(frame, (display_width, display_height))

        # Afficher l'image traitée
        cv2.imshow('Processed Video', warped_image)
        
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
