import cv2
import numpy as np

def detect_chessboard_corners(image, show_process=False):
    """
    Détecte les coins du plateau d'échecs (a1, a8, h1, h8) dans l'image donnée.
    
    :param image: Image d'entrée (BGR)
    :param show_process: Booléen pour afficher les étapes intermédiaires
    :return: Liste des coordonnées des coins [a1, a8, h1, h8] ou None si non détecté
    """
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if show_process:
        cv2.imshow('Blurred Image', blurred)
        cv2.waitKey(100)
    
    # * Détecter les coins avec l'algorithme de Harris
    # corners = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)

    # # Normaliser les coins pour une meilleure visualisation
    # corners_normalized = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # if show_process:
    #     cv2.imshow('Corners', corners_normalized)
    #     cv2.waitKey(100)
    
    # # Dilater les coins pour les rendre plus visibles
    # corners_dilated = cv2.dilate(corners, None)
    
    # if show_process:
    #     corners_dilated_normalized = cv2.normalize(corners_dilated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     cv2.imshow('Dilated Corners', corners_dilated_normalized)
    #     cv2.waitKey(100)
    
    # # Seuiller pour obtenir uniquement les coins les plus forts
    # threshold = 0.01 * corners_dilated.max()
    # corner_points = np.where(corners_dilated > threshold)
    
    # if show_process:
    #     thresholded = np.zeros_like(gray)
    #     thresholded[corner_points] = 255
    #     cv2.imshow('Thresholded Corners', thresholded)
    #     cv2.waitKey(100)


    # * Détecter les bords avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    if show_process:
        cv2.imshow('Edges', edges)
        cv2.waitKey(100)

    # * Utiliser la transformation de Hough pour détecter les lignes
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    # lines = detect_lines_with_adaptive_threshold(edges)


    if lines is None or len(lines) < 10:  # Nous avons besoin d'au moins quelques lignes
        return None

    # * Filtrer les lignes pour ne garder que les horizontales et verticales

    horizontal_lines, vertical_lines = filter_lines(lines)

    # horizontal_lines = []
    # vertical_lines = []
    # for line in lines:
    #     rho, theta = line[0]
    #     if np.abs(theta) < np.pi/4 or np.abs(theta) > 3*np.pi/4:
    #         vertical_lines.append(line)
    #     else:
    #         horizontal_lines.append(line)
    
    if show_process:
        line_image = image.copy()
        draw_lines(line_image, horizontal_lines, (0, 255, 0))  # Lignes horizontales en vert
        draw_lines(line_image, vertical_lines, (0, 0, 255))    # Lignes verticales en rouge
        cv2.imshow('Detected Lines', line_image)
        cv2.waitKey(100)
    
    # * Trouver les intersections des lignes
    intersections = find_intersections(horizontal_lines, vertical_lines)
    
    if len(intersections) < 4:
        return None
    
    # * Trouver les coins du plateau
    corners = find_chessboard_corners(intersections)
    
    if show_process:
        corner_image = image.copy()
        for corner in corners:
            cv2.circle(corner_image, tuple(map(int, corner)), 5, (0, 255, 255), -1)
        cv2.imshow('Detected Corners', corner_image)
        cv2.waitKey(100)
    
    return corners
    
    # Dilater les bords pour fermer les contours
    # kernel = np.ones((3,3), np.uint8)
    # dilated = cv2.dilate(edges, kernel, iterations=1)

    # if show_process:
    #     cv2.imshow('Dilated Edges', dilated)
    #     cv2.waitKey(100)


    # Trouver les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trier les contours par aire décroissante
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours[:5]:  # Vérifier les 5 plus grands contours
        # Approximer le contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Si le contour a 4 côtés, c'est probablement notre plateau
        if len(approx) == 4:
            if show_process:
                contour_image = image.copy()
                cv2.drawContours(contour_image, [approx], 0, (0, 255, 0), 2)
                cv2.imshow('Detected Chessboard', contour_image)
                cv2.waitKey(100)
            
            # Trier les points pour obtenir [a1, a8, h8, h1]
            corners = sort_rectangle_points(approx.reshape(4, 2))
            return corners
    
    return None
    
    # Convertir en liste de points (x, y)
    points = np.array(list(zip(corner_points[1], corner_points[0])))
    
    if len(points) < 4:
        return None  # Pas assez de coins détectés
    
    # Trouver le rectangle englobant
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Trier les points pour obtenir [a1, a8, h8, h1]
    box = sort_rectangle_points(box)
    
    return box



# def detect_lines_with_adaptive_threshold(edges):
#     """Détecte les lignes avec un seuil adaptatif."""
#     lines = []
#     threshold = 100
#     min_lines = 20
#     max_lines = 300

#     while len(lines) < min_lines and threshold > 50:
#         lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
#         if lines is None:
#             lines = []
#         threshold -= 10

#     while len(lines) > max_lines and threshold < 200:
#         lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
#         if lines is None:
#             lines = []
#         threshold += 10

#     return lines



def filter_lines(lines):
    """Filtre les lignes pour ne garder que les 9 horizontales et 9 verticales les plus pertinentes."""
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        if np.abs(theta) < np.pi/4 or np.abs(theta) > 3*np.pi/4:
            vertical_lines.append((rho, theta))
        else:
            horizontal_lines.append((rho, theta))


    # Fonction pour regrouper les lignes proches
    def group_lines(lines, max_distance=20):
        groups = []
        for line in lines:
            rho, theta = line
            if not groups or abs(rho - groups[-1][-1][0]) > max_distance:
                groups.append([(rho, theta)])
            else:
                groups[-1].append((rho, theta))
        return [(sum(rho for rho, _ in group) / len(group),
                 sum(theta for _, theta in group) / len(group)) for group in groups]

    # Regrouper les lignes proches et prendre la moyenne de chaque groupe
    horizontal_lines = group_lines(horizontal_lines)
    vertical_lines = group_lines(vertical_lines)

    # Fonction pour trouver les lignes les plus parallèles et régulièrement espacées
    def find_best_lines(lines, n=9):
        if len(lines) < n:
            return lines
        
        lines = sorted(lines, key=lambda l: l[0])  # Trier par rho
        best_score = float('inf')
        best_lines = []
        
        for i in range(len(lines) - n + 1):
            candidate_lines = lines[i:i+n]
            rhos = [l[0] for l in candidate_lines]
            thetas = [l[1] for l in candidate_lines]
            
            # Calculer la régularité de l'espacement
            rho_diffs = np.diff(rhos)
            spacing_score = np.std(rho_diffs)
            
            # Calculer le parallélisme
            parallelism_score = np.std(thetas)
            
            # Score combiné
            score = spacing_score + parallelism_score
            
            if score < best_score:
                best_score = score
                best_lines = candidate_lines
        
        return best_lines

    horizontal_lines = find_best_lines(horizontal_lines)
    vertical_lines = find_best_lines(vertical_lines)

    return horizontal_lines, vertical_lines

def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            rho1, theta1 = h_line
            rho2, theta2 = v_line
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            intersections.append((x0[0], y0[0]))
    return intersections

def find_chessboard_corners(intersections):
    # Trier les intersections par leur distance au coin supérieur gauche de l'image
    sorted_intersections = sorted(intersections, key=lambda p: p[0]**2 + p[1]**2)
    
    # Prendre les 4 coins les plus extrêmes
    corners = sorted_intersections[:4]
    
    # Trier les coins pour obtenir [a1, a8, h8, h1]
    return sort_rectangle_points(corners)

def sort_rectangle_points(points):
    """
    Trie les points d'un rectangle pour obtenir [a1, a8, h8, h1].
    
    :param points: Liste de 4 points [(x, y), ...]
    :return: Liste triée [a1, a8, h8, h1]
    """
    # Trier par la somme des coordonnées (x+y)
    sorted_points = sorted(points, key=lambda p: p[0] + p[1])
    
    # a1 est le point avec la plus grande somme (en bas à gauche)
    a1 = sorted_points[3]
    
    # h8 est le point avec la plus petite somme (en haut à droite)
    h8 = sorted_points[0]
    
    # Pour différencier a8 et h1, on compare leurs coordonnées x
    if sorted_points[1][0] < sorted_points[2][0]:
        a8, h1 = sorted_points[1], sorted_points[2]
    else:
        a8, h1 = sorted_points[2], sorted_points[1]
    
    return [a1, a8, h8, h1]

def draw_lines(image, lines, color):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

def draw_chessboard_corners(image, corners):
    """
    Dessine les coins détectés sur l'image.
    
    :param image: Image d'entrée
    :param corners: Liste des coordonnées des coins [a1, a8, h8, h1]
    :return: Image avec les coins dessinés
    """
    if corners is None:
        return image
    
    # Convertir les coins en entiers
    corners = np.int32(corners)
    
    # Dessiner les coins
    for i, corner in enumerate(corners):
        cv2.circle(image, tuple(corner), 5, (0, 0, 255), -1)
        cv2.putText(image, ['a1', 'a8', 'h8', 'h1'][i], tuple(corner + [10, 10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Dessiner le contour du plateau
    cv2.polylines(image, [corners], True, (0, 255, 0), 2)
    
    return image



# * Test du code
if __name__ == "__main__":

    frame_interval = 1000
    frame_count = 0

    # Charger une vidéo de test
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur: Impossible de charger la vidéo {video_path}")
    else:
        while True:

            if frame_count % frame_interval == 0:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()

                if not ret:
                    break

                # Détecter les coins du plateau
                corners = detect_chessboard_corners(frame, show_process=True)
                
                # if corners is not None:
                #     # Dessiner les coins sur l'image
                #     result = draw_chessboard_corners(frame.copy(), corners)
                    
                    # Afficher le résultat
                    # cv2.imshow("Coins du plateau d'échecs", result)
                # else:
                #     cv2.imshow("Vidéo originale", frame)

                # Attendre 1ms entre chaque frame et vérifier si l'utilisateur veut quitter
                key = cv2.waitKey(100) & 0xFF

                if key == ord('q'):
                    print("Fermeture des fenêtres")
                    break

            frame_count += 1

        cap.release()

