import cv2
import numpy as np
from sticker_calibration import detect_stickers, draw_stickers
from polygon import detect_chessboard, draw_chessboard
# from camera_calibration import calibrate_camera, undistort_frame



def process_video(video_path):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Vérifier si la vidéo est ouverte correctement
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
        print(f"Code d'erreur : {cap.get(cv2.CAP_PROP_FOURCC)}")
        return
    
    # Obtenir les propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # * Traiter l'image
        # blue_stickers, pink_stickers, chessboard_corners, approx = process_frame(frame)

        # Appliquer la détection des autocollants
        blue_stickers, pink_stickers = detect_stickers(frame)
        
        # Appliquer la détection de l'échiquier
        chessboard_corners, approx = detect_chessboard(frame)


        # Calibrer la caméra
        # chessboard_size = (7, 7)
        # frame_size = (640, 480)
        # camera_matrix, dist_coeffs = calibrate_camera(frame, chessboard_size, frame_size)

        # Appliquer la correction de distorsion

        # if camera_matrix is not None and dist_coeffs is not None:
        #     undistorted_image = undistort_frame(frame, camera_matrix, dist_coeffs)
        #     cv2.imshow('Undistorted Image', undistorted_image)
        
        # * Dessiner les résultats


        frame = draw_stickers(frame, blue_stickers, pink_stickers)
    
        # Dessiner l'échiquier s'il est détecté
        if chessboard_corners is not None:
            frame = draw_chessboard(frame, chessboard_corners, approx)

        # frame_with_results = draw_results(frame, blue_stickers, pink_stickers, chessboard_corners, approx)
        
        # Afficher l'image traitée
        # cv2.imshow('Processed Video', frame)
        
        # Attendre 1ms entre chaque frame et vérifier si l'utilisateur veut quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    process_video(video_path)

# Note: Assurez-vous que les fonctions importées (detect_stickers, detect_chessboard, calibrate_camera)
# sont adaptées pour fonctionner sur une seule image à la fois.
