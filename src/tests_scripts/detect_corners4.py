import cv2
import numpy as np


def detect_chessboard_corners(img, show_process=False):
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

    cornersFound, corners = cv2.findChessboardCorners(small_gray, chessboard_size,
                                                      cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                      cv2.CALIB_CB_FAST_CHECK +
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)

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

        # print(cornersRefined)
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

        a1 = top_left + (cornersRefined[0][0] - cornersRefined[1][0]) + (
                    cornersRefined[0][0] - cornersRefined[chessboard_size[0]][0])
        a8 = bottom_left + (cornersRefined[-chessboard_size[0]][0] - cornersRefined[-chessboard_size[0] + 1][0]) + (
                    cornersRefined[-chessboard_size[0]][0] - cornersRefined[-chessboard_size[0] * 2][0])
        h1 = bottom_right + (cornersRefined[-1][0] - cornersRefined[-2][0]) + (
                    cornersRefined[-1][0] - cornersRefined[-chessboard_size[0] - 1][0])
        h8 = top_right + (cornersRefined[chessboard_size[0] - 1][0] - cornersRefined[chessboard_size[0] - 2][0]) + (
                    cornersRefined[chessboard_size[0] - 1][0] - cornersRefined[chessboard_size[0] * 2 - 1][0])

        corners_extremities = [a1, a8, h1, h8]
        # print("Extremities of the chessboard corners:", corners_extremities)

        if show_process:
            # Optionnel : afficher les coins détectés
            cv2.imshow('Chessboard Detection', img)
            cv2.waitKey(500)

        return corners_extremities

    else:
        if show_process:
            cv2.imshow('Chessboard Detection', img)

    # Si aucun échiquier n'est détecté, retourner None, None
    return None


def refine_corners(img, chessboard_corners, search_radius=20, show_process=False):
    """
    Polishes the corners to improve their accuracy.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # * Détecter les coins avec l'algorithme de Harris
    corners = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)

    # # Normaliser les coins pour une meilleure visualisation
    corners_normalized = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # if show_process:
    #     cv2.imshow('Corners', corners_normalized)
    #     cv2.waitKey(100)

    # # Dilater les coins pour les rendre plus visibles
    corners_dilated = cv2.dilate(corners, None)

    # if show_process:
    #     corners_dilated_normalized = cv2.normalize(corners_dilated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     cv2.imshow('Dilated Corners', corners_dilated_normalized)
    #     cv2.waitKey(100)

    # # Seuiller pour obtenir uniquement les coins les plus forts
    threshold = 0.001 * corners_dilated.max()
    corner_points = np.where(corners_dilated > threshold)

    # if show_process:
    #     thresholded = img
    #     thresholded[corner_points] = 0

    #     for i, corner in enumerate(chessboard_corners):
    #         corner_int = tuple(corner.astype(int))
    #         cv2.circle(gray, corner_int, 5, (0, 0, 255), -1)

    #     cv2.imshow('Thresholded Corners', thresholded)
    #     cv2.waitKey(100)

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

    # if show_process:
    #     # Afficher les coins raffinés
    #     corner_img = img.copy()
    #     for i, (original, refined) in enumerate(zip(chessboard_corners, refined_corners)):
    #         cv2.circle(corner_img, tuple(map(int, original)), 5, (0, 0, 255), -1)  # Original en rouge
    #         cv2.circle(corner_img, refined, 5, (0, 255, 0), -1)  # Raffiné en vert
    #         cv2.line(corner_img, tuple(map(int, original)), refined, (255, 0, 0), 2)  # Ligne bleue entre les deux
    #         # cv2.putText(corner_img, f'Corner {i+1}', refined, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #     cv2.imshow('Refined Corners', corner_img)
    #     cv2.waitKey(0)

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


def detect_stickers(img, corners, distance_threshold=100):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

    # range for blue sticker
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([150, 255, 255])

    # range for pink sticker
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])

    # threshold to get only blue and pink colors
    mask_blue = cv2.inRange(blurred, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)

    mask_pink = cv2.inRange(blurred, lower_pink, upper_pink)

    blue_stickers = None
    pink_stickers = None

    labeled_corners = {'a1': None, 'a8': None, 'h8': None, 'h1': None}

    # Detect blue stickers
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)
    if contours_blue:
        for c in contours_blue:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
            blue_stickers = (int(cX), int(cY), int(radius))

            labeled_corners['a1'] = tuple(sorted_corners[0])
            labeled_corners['h1'] = tuple(sorted_corners[1])
            break

    # Detect pink stickers
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink = sorted(contours_pink, key=cv2.contourArea, reverse=True)
    if contours_pink:
        for c in contours_pink:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
            pink_stickers = (int(cX), int(cY), int(radius))

            labeled_corners['a8'] = tuple(sorted_corners[0])
            labeled_corners['h8'] = tuple(sorted_corners[1])
            break

    return blue_stickers, pink_stickers, labeled_corners


def draw_refined_corners(img, original_corners, refined_corners, blue_stickers, pink_stickers, search_radius=20):
    corner_img = img.copy()

    for i, (original, refined) in enumerate(zip(original_corners, refined_corners)):
        # Draw search radius circle around original corner
        cv2.circle(corner_img, tuple(map(int, original)), search_radius, (255, 255, 0),
                   1)  # Yellow circle for search radius

        cv2.circle(corner_img, tuple(map(int, original)), 5, (0, 0, 255), -1)  # Original en rouge
        cv2.circle(corner_img, refined, 5, (0, 255, 0), -1)  # Raffiné en vert
        cv2.line(corner_img, tuple(map(int, original)), refined, (255, 0, 0), 2)  # Ligne bleue entre les deux
        # cv2.putText(corner_img, f'Corner {i+1}', refined, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for (cX, cY, radius) in blue_stickers:
        cv2.circle(corner_img, (cX, cY), radius, (255, 0, 0), 2)  # Blue outline
        cv2.circle(corner_img, (cX, cY), 3, (0, 255, 0), -1)  # Center in green

    # Draw pink stickers
    for (cX, cY, radius) in pink_stickers:
        cv2.circle(corner_img, (cX, cY), radius, (255, 0, 255), 2)  # Pink outline
        cv2.circle(corner_img, (cX, cY), 3, (0, 255, 0), -1)  # Center in green

    # cv2.imshow('Refined Corners', corner_img)q
    # cv2.waitKey(0)
    return corner_img


def draw_chessboard(img, corners, blue_stickers=None, pink_stickers=None):
    """
    Dessine l'échiquier détecté sur l'image en utilisant polylines.
    
    :param img: Image d'entrée
    :param corners: Liste des coordonnées des coins [a1, a8, h8, h1]
    :param blue_stickers: Liste des positions des stickers bleus (optionnel) au format (x, y, rayon)
    :param pink_stickers: Liste des positions des stickers roses (optionnel) au format (x, y, rayon)
    :return: Image avec l'échiquier dessiné
    """
    if corners is None:
        return img

    # Convertir la liste de coins en tableau numpy
    corners_array = np.array(corners, dtype=np.int32)
    
    # Dessiner le contour de l'échiquier
    cv2.polylines(img, [corners_array], True, (0, 255, 0), 2)
    
    # Dessiner les coins individuels avec leurs labels
    corner_labels = ['a1', 'a8', 'h8', 'h1']
    for i, corner in enumerate(corners):
        corner_int = tuple(corner.astype(int))
        cv2.circle(img, corner_int, 5, (0, 0, 255), -1)
        cv2.putText(img, corner_labels[i], (corner_int[0] + 10, corner_int[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Dessiner les stickers si présents
    if blue_stickers is not None and len(blue_stickers) > 0:
        cX, cY, radius = blue_stickers[0]  # Déballage du tuple
        cv2.circle(img, (cX, cY), radius, (255, 0, 0), 2)  # Contour du sticker
        cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Centre du sticker

    if pink_stickers is not None and len(pink_stickers) > 0:
        cX, cY, radius = pink_stickers[0]  # Déballage du tuple
        cv2.circle(img, (cX, cY), radius, (255, 0, 255), 2)  # Contour du sticker
        cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Centre du sticker
    
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

                    img_with_chessboard = draw_chessboard(frame.copy(), corners, blue_stickers, pink_stickers)

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


