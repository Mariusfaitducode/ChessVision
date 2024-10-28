import cv2
import numpy as np

from threading import Thread
from queue import Queue

from stickers_detection import detect_stickers, draw_stickers


def detect_chessboard_corners(img):
    """
    Détecte l'échiquier dans l'image et retourne ses coins.

    :param img: Image d'entrée en couleur (BGR)
    :param show_process: Booléen pour afficher les étapes intermédiaires
    :return: Liste des coordonnées des coins [a1, a8, h8, h1] et contour approché, ou (None, None) si non détecté
    """

    chessboard_size = (7, 7)

    # Conversion de l'image en niveaux de gris pour simplifier le traitement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    start_time = cv2.getTickCount()

    # Réduire la résolution de l'image pour accélérer la détection
    scale = 0.5
    small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    
    cornersFound, corners = find_corners_with_timeout(
        small_gray, 
        chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS
    )

    if cornersFound:
        # Ajuster les coordonnées des coins à l'échelle originale
        corners = corners * (1.0 / scale)
    else:
        corners = None

    end_time = cv2.getTickCount()
    execution_time = (end_time - start_time) / cv2.getTickFrequency()
    print(f"Temps d'exécution de findChessboardCorners: {execution_time:.4f} secondes")

    if cornersFound:
        objpoints.append(objp)
        cornersRefined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(cornersRefined)

        # cv2.drawChessboardCorners(img, chessboard_size, cornersRefined, cornersFound)

        # Retrieve the 4 points at the extremities of the matrix in cornersRefined
        top_left = cornersRefined[0][0]
        top_right = cornersRefined[chessboard_size[0] - 1][0]
        bottom_right = cornersRefined[-1][0]
        bottom_left = cornersRefined[-chessboard_size[0]][0]

        # Retrouver les 4 points extremities de l'échiquier
        # square_size = np.linalg.norm(cornersRefined[0] - cornersRefined[chessboard_size[0]])

        # top_left -= [square_size, square_size]
        # top_right += [square_size, square_size]
        # bottom_right += [square_size, square_size]
        # bottom_left -= [square_size, square_size]

        # TODO : clean this code

        a1 = top_left + (cornersRefined[0][0] - cornersRefined[1][0]) + (
                    cornersRefined[0][0] - cornersRefined[chessboard_size[0]][0])
        a8 = bottom_left + (cornersRefined[-chessboard_size[0]][0] - cornersRefined[-chessboard_size[0] + 1][0]) + (
                    cornersRefined[-chessboard_size[0]][0] - cornersRefined[-chessboard_size[0] * 2][0])
        h1 = bottom_right + (cornersRefined[-1][0] - cornersRefined[-2][0]) + (
                    cornersRefined[-1][0] - cornersRefined[-chessboard_size[0] - 1][0])
        h8 = top_right + (cornersRefined[chessboard_size[0] - 1][0] - cornersRefined[chessboard_size[0] - 2][0]) + (
                    cornersRefined[chessboard_size[0] - 1][0] - cornersRefined[chessboard_size[0] * 2 - 1][0])

        corners_extremities = [a1, a8, h1, h8]

        return corners_extremities

    # Si aucun échiquier n'est détecté, retourner None
    return None



def find_corners_with_timeout(gray, chessboard_size, flags):
    result_queue = Queue()
    
    def worker():
        result = cv2.findChessboardCorners(gray, chessboard_size, flags)
        result_queue.put(result)
    
    thread = Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Attendre max 0.05 secondes
    thread.join(timeout=0.05)
    
    if result_queue.empty():
        return False, None
    
    return result_queue.get()


def refine_corners(img, chessboard_corners, search_radius=20):
    """
    Polishes the corners to improve their accuracy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # * Détecter les coins avec l'algorithme de Harris
    corners = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)

    # # Normaliser les coins pour une meilleure visualisation
    # corners_normalized = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # # Dilater les coins pour les rendre plus visibles
    corners_dilated = cv2.dilate(corners, None)


    # # Seuiller pour obtenir uniquement les coins les plus forts
    threshold = 0.001 * corners_dilated.max()
    corner_points = np.where(corners_dilated > threshold)


    # Trouver les coins les plus proches pour chaque coin de l'échiquier
    refined_corners = []
    for corner in chessboard_corners:
        # Convertir le coin en coordonnées entières
        corner_int = tuple(map(int, corner))

        # Définir une région de recherche autour du coin
        y_min = max(0, corner_int[1] - search_radius)
        y_max = min(img.shape[0], corner_int[1] + search_radius)
        x_min = max(0, corner_int[0] - search_radius)
        x_max = min(img.shape[1], corner_int[0] + search_radius)

        # Trouver tous les coins dans la région de recherche
        region_corners = np.argwhere(corners_dilated[y_min:y_max, x_min:x_max] > threshold)

        if len(region_corners) > 0:
            # Calculer les distances au coin original
            distances = np.sum((region_corners - [corner_int[1] - y_min, corner_int[0] - x_min]) ** 2, axis=1)

            # Trouver le coin le plus proche
            closest_corner = region_corners[np.argmin(distances)]
            refined_corner = (x_min + closest_corner[1], y_min + closest_corner[0])
        else:
            # Si aucun coin n'est trouvé, garder le coin original
            refined_corner = corner_int

        refined_corners.append(refined_corner)

    return refined_corners


def get_warped_image(img, corners):
    # Ensure corners are in the correct order: [top-left, top-right, bottom-right, bottom-left]
    src_pts = np.array(corners, dtype=np.float32)

    # Define the size of the output image (e.g., 800x800 pixels)
    width, height = 800, 800
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped


def draw_refined_corners(img, original_corners, refined_corners, search_radius=20):
    corner_img = img.copy()

    if original_corners is None:

        for i, refined in enumerate(refined_corners):
            cv2.circle(corner_img, refined, 5, (0, 255, 0), -1)  # Raffiné en vert

    else:

        for i, (original, refined) in enumerate(zip(original_corners, refined_corners)):
            # Draw search radius circle around original corner
            cv2.circle(corner_img, tuple(map(int, original)), search_radius, (255, 255, 0), 1)  # Yellow circle for search radius

            cv2.circle(corner_img, tuple(map(int, original)), 5, (0, 0, 255), -1)  # Original en rouge
            cv2.circle(corner_img, refined, 5, (0, 255, 0), -1)  # Raffiné en vert
            cv2.line(corner_img, tuple(map(int, original)), refined, (255, 0, 0), 2)  # Ligne bleue entre les deux
            # cv2.putText(corner_img, f'Corner {i+1}', refined, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    

    # cv2.imshow('Refined Corners', corner_img)
    # cv2.waitKey(0)
    return corner_img


def draw_labeled_chessboard(img, labeled_corners):
   
    if labeled_corners is None:
        return img

    # Draw labels for each corner
    for label, coords in labeled_corners.items():
        if coords is not None:
            # Convert coordinates to integers and tuple
            coords = tuple(map(int, coords))
            # Draw text slightly offset from the corner point
            cv2.putText(img, label, 
                       (coords[0] + 10, coords[1] + 10),  # Offset text position
                       cv2.FONT_HERSHEY_SIMPLEX,  # Font
                       0.8,  # Font scale
                       (0, 0, 0),  # Color (green)
                       2)  # Thickness

    
    
    return img


# Cette partie ne s'exécutera que si le script est exécuté directement (pas importé)
if __name__ == "__main__":
    # Tester la fonction avec une seule image
    img = cv2.imread('src/calibration_images/img1.png')

    # Charger une vidéo de test
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    cap = cv2.VideoCapture(video_path)

    frame_interval = 100
    frame_count = 0

    # last_area = None

    if not cap.isOpened():
        print(f"Erreur: Impossible de charger la vidéo {video_path}")
    else:
        while True:

            if frame_count % frame_interval == 0:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()

                if not ret:
                    break

                corners = detect_chessboard_corners(frame, show_process=False)

                if corners is not None:

                    print("CORNERS : ", corners)

                    refine_corners(frame, corners, show_process=True)

                    blue_sticker, pink_sticker, labeled_corners = detect_stickers(frame, corners)
                    blue_stickers = [blue_sticker] if blue_sticker else []
                    pink_stickers = [pink_sticker] if pink_sticker else []

                    img_with_chessboard = draw_labeled_chessboard(frame.copy(), corners, blue_stickers, pink_stickers)

                    # display_width = 800  # Vous pouvez ajuster cette valeur selon vos besoins
                    # aspect_ratio = img_with_chessboard.shape[1] / img_with_chessboard.shape[0]
                    # display_height = int(display_width / aspect_ratio)
                    # display_frame = cv2.resize(img_with_chessboard, (display_width, display_height))

                    # Afficher l'image traitée
                    # cv2.imshow('Processed Video', display_frame)

                    cv2.imshow("Detected Chessboard", img_with_chessboard)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    print("No chessboard detected")

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Fermeture des fenêtres")
                    break

            frame_count += 1

        cap.release()

