import numpy as np
import cv2
from tqdm import tqdm

from utils import find_corners_with_timeout, resize_frame


def compute_homography(frame, board_size=(7, 7), square_size=100):
    """
    Calcule la matrice d'homographie et la pose 3D pour une image donnée.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Réduire la résolution de l'image pour accélérer la détection
    scale = 0.5
    small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    cornersFound, corners = find_corners_with_timeout(
        small_gray, 
        board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS
    )

    if not cornersFound:
        return None, None, None
    else:
        corners = corners * (1.0 / scale)

        # Affiner la détection des coins
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        

        # Points 3D de l'échiquier
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size


        # Calculer l'homographie
        dst_points = np.zeros((board_size[0] * board_size[1], 2), np.float32)
        for i in range(board_size[0]):
            for j in range(board_size[1]):
                dst_points[i*board_size[1] + j] = [j*square_size, i*square_size]
        
        H, _ = cv2.findHomography(corners.reshape(-1, 2), dst_points)
        
        return H, corners,objp
    


def compute_pose(objp, labeled_corners):
    # Convert labeled corners to ordered list of chess corners
    # Order: a1, a8, h8, h1
    if 'a1' in labeled_corners and 'a8' in labeled_corners and 'h8' in labeled_corners and 'h1' in labeled_corners:
        chess_corners = np.array([
            labeled_corners['a1'],
            labeled_corners['a8'], 
            labeled_corners['h8'],
            labeled_corners['h1']
        ], dtype=np.float32)
        
        # Créer les points 3D correspondants aux 4 coins
        # Pour un échiquier 7x7 avec des carrés de 100mm
        object_points = np.array([
            [0, 0, 0],           # a1 (coin bas gauche)
            [0, 600, 0],         # a8 (coin haut gauche)
            [600, 600, 0],       # h8 (coin haut droit)
            [600, 0, 0]          # h1 (coin bas droit)
        ], dtype=np.float32)
    else:
        return None, None

    # Paramètres de la caméra
    calibration_results = np.load('camera_calibration_results.npz')
    camera_matrix = calibration_results['cameraMatrix']
    dist_coeffs = calibration_results['distCoeffs']

    ret, rvec, tvec = cv2.solvePnP(object_points, chess_corners.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
    return rvec, tvec



def draw_axis(frame, rvec, tvec):
    """
    Dessine les axes 3D sur l'image.
    """
    # Paramètres de la caméra (estimation approximative si non calibrée)
    camera_matrix = np.array([[frame.shape[1], 0, frame.shape[1]/2],
                             [0, frame.shape[1], frame.shape[0]/2],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))

    # Points pour dessiner les axes
    axis_length = 100
    axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
    
    # Projeter les points des axes
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    # Dessiner les axes
    origin = tuple(imgpts[0].ravel())
    frame = cv2.line(frame, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)  # X axis (Rouge)
    frame = cv2.line(frame, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)  # Y axis (Vert)
    frame = cv2.line(frame, origin, tuple(imgpts[3].ravel()), (255,0,0), 3)  # Z axis (Bleu)
    
    return frame



def display_results(frame, H, corners, rvec, tvec, board_size=(7, 7), output_size=(700, 700)):
    """
    Affiche l'image originale avec les coins détectés, les axes 3D et l'image transformée.
    """
    # Dessiner les coins et les axes
    frame_with_corners = frame.copy()
    cv2.drawChessboardCorners(frame_with_corners, board_size, corners, True)
    frame_with_corners = draw_axis(frame_with_corners, rvec, tvec)
    
    # Appliquer la transformation homographique
    warped_frame = cv2.warpPerspective(frame, H, output_size)
    
    return frame_with_corners, warped_frame




if __name__ == "__main__":
    # Charger et traiter la vidéo
    video_path = 'videos/moving_game.MOV'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculer l'homographie et la pose
        H, corners, objp = compute_homography(frame)
        
        if H is not None:


            rvec, tvec = compute_pose(objp, corners)


            # Afficher les résultats
            new_frame, warped_frame = display_results(frame, H, corners, rvec, tvec)


            # Dessiner les points source
            # for point in src_points:
            #     cv2.circle(new_frame, tuple(map(int, point)), 10, (0, 0, 255), -1)


            # Redimensionner pour l'affichage
            # 
            
            frame_resized = resize_frame(new_frame, 800)
            warped_resized = resize_frame(warped_frame, 800)

            

            cv2.imshow('Detected Corners and Axes', frame_resized)
            # cv2.imshow('Warped Chessboard', warped_resized)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            # print("No chessboard detected")

            display_width = 800
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_height = int(display_width / aspect_ratio)

            frame_resized = cv2.resize(frame, (display_width, display_height))

            cv2.imshow('Detected Corners and Axes', frame_resized)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        pbar.update(1)
        

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
