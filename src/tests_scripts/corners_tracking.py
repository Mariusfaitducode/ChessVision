import cv2
import numpy as np

from utils import *


def estimate_corners_movement(prev_grid, curr_frame, prev_frame, debug=False):
    """
    Estime le mouvement des coins entre deux frames en utilisant le flux optique.
    
    Args:
        prev_grid: Grille précédente (9x9x2) contenant tous les coins
        curr_frame: Frame courante
        prev_frame: Frame précédente
        debug: Afficher la visualisation du tracking
    
    Returns:
        Nouvelle grille (9x9x2) avec les positions estimées des coins
    """
    if prev_frame is None or prev_grid is None:
        return None, None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Convertir la grille 9x9 en liste de points
    prev_corners_array = prev_grid.copy().reshape(-1, 1, 2).astype(np.float32)
    
    # Calculer le flux optique
    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_corners_array, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    if np.all(status == 1):
        new_grid = next_corners.reshape(9, 9, 2)
        valid_mask = validate_tracked_corners(prev_grid, new_grid, curr_frame, prev_frame)
        
        debug_frame = curr_frame.copy() if debug else None
        if debug_frame is not None:
            # Dessiner les points valides en vert
            for i in range(9):
                for j in range(9):

                    # prev_x, prev_y = map(int, prev_grid[i, j])
                    # cv2.circle(debug_frame, (prev_x, prev_y), 3, (0, 0, 0), -1)

                    if valid_mask[i, j]:
                        x, y = map(int, new_grid[i, j])
                        cv2.circle(debug_frame, (x, y), 5, (0, 0, 0), -1)

        # Attendre l'appui sur espace pour continuer ou q pour quitter
        if debug:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Espace
                    break
                elif key == ord('q'):  # q
                    cv2.destroyAllWindows()
                    exit()
        
        # Reconstruire la grille
        reconstructed_grid = reconstruct_grid(new_grid, valid_mask, debug_frame)
        
        if debug:
            cv2.imshow('Reconstructed Grid', debug_frame)
            cv2.waitKey(1)


        extremities = [
            reconstructed_grid[0, 0],  # a8 (top-left)
            reconstructed_grid[8, 0],  # a1 (bottom-left)
            reconstructed_grid[8, 8],  # h1 (bottom-right)
            reconstructed_grid[0, 8],  # h8 (top-right)
        ]
        
        return reconstructed_grid, extremities, None
    
    return None, None, None



def validate_tracked_corners(prev_grid, new_grid, curr_frame, prev_frame, max_movement_ratio=2):
    """
    Valide les coins trackés et retourne un masque des coins valides.
    
    Args:
        prev_grid: Grille précédente (9x9x2)
        new_grid: Nouvelle grille estimée (9x9x2)
        curr_frame: Frame courante
        prev_frame: Frame précédente
        max_movement_ratio: Ratio maximum de mouvement par rapport à la médiane
    
    Returns:
        Masque booléen (9x9) indiquant les coins valides (True) et invalides (False)
    """
    valid_mask = np.ones((9, 9), dtype=bool)
    
    # 1. Validation par cohérence de mouvement
    movement_vectors = new_grid - prev_grid
    movements = np.linalg.norm(movement_vectors.reshape(-1, 2), axis=1)
    median_movement = np.median(movements)
    
    # Les mouvements trop grands sont suspects
    movement_mask = movements.reshape(9, 9) < (median_movement * max_movement_ratio)
    valid_mask &= movement_mask
    
    # 2. Validation par intensité locale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    window_size = 5
    intensity_threshold = 50  # À ajuster
    
    for i in range(9):
        for j in range(9):
            if valid_mask[i, j]:
                # Extraire les fenêtres autour du point dans les deux frames
                x, y = map(int, new_grid[i, j])
                if 0 <= x < curr_gray.shape[1] - window_size and 0 <= y < curr_gray.shape[0] - window_size:
                    curr_window = curr_gray[y:y+window_size, x:x+window_size]
                    
                    prev_x, prev_y = map(int, prev_grid[i, j])
                    if 0 <= prev_x < prev_gray.shape[1] - window_size and 0 <= prev_y < prev_gray.shape[0] - window_size:
                        prev_window = prev_gray[prev_y:prev_y+window_size, prev_x:prev_x+window_size]
                        
                        # Si l'apparence locale a trop changé, le point est suspect
                        if np.abs(np.mean(curr_window) - np.mean(prev_window)) > intensity_threshold:
                            valid_mask[i, j] = False
    
    # 3. Validation par détection de coins Harris
    corners_harris = cv2.cornerHarris(curr_gray, blockSize=2, ksize=3, k=0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    
    search_radius = 10
    corner_threshold = 0.01 * corners_harris.max()
    
    for i in range(9):
        for j in range(9):
            if valid_mask[i, j]:
                x, y = map(int, new_grid[i, j])
                if 0 <= x < curr_gray.shape[1] and 0 <= y < curr_gray.shape[0]:
                    # Vérifier si un coin fort existe dans le voisinage
                    y_min = max(0, y - search_radius)
                    y_max = min(curr_gray.shape[0], y + search_radius)
                    x_min = max(0, x - search_radius)
                    x_max = min(curr_gray.shape[1], x + search_radius)
                    
                    region = corners_harris[y_min:y_max, x_min:x_max]
                    if region.max() < corner_threshold:
                        valid_mask[i, j] = False
    
    return valid_mask





def reconstruct_grid(new_grid, valid_mask, debug_frame=None):
    reconstructed_grid = new_grid.copy()
    reconstruction_mask = valid_mask.copy()
    point_weights = valid_mask.astype(float)
    
    directions = [
        [(-1, 0), (-2, 0), (-3, 0)],  # gauche
        [(1, 0), (2, 0), (3, 0)],     # droite
        [(0, -1), (0, -2), (0, -3)],  # haut
        [(0, 1), (0, 2), (0, 3)],     # bas
        [(-1, -1), (-2, -2), (-3, -3)],  # diagonale haut-gauche
        [(1, -1), (2, -2), (3, -3)],     # diagonale haut-droite
        [(-1, 1), (-2, 2), (3, 3)],      # diagonale bas-gauche
        [(1, 1), (2, 2), (3, 3)]         # diagonale bas-droite
    ]
    
    points_added = True
    while points_added:
        points_added = False
        
        for i in range(9):
            for j in range(9):
                if not reconstruction_mask[i, j]:
                    ratio_interpolations = []
                    simple_interpolations = []
                    fallback_interpolations = []
                    debug_lines = []  # Pour stocker les lignes à dessiner
                    
                    for dir_pair in directions:
                        p1_i, p1_j = i + dir_pair[0][0], j + dir_pair[0][1]
                        p2_i, p2_j = i + dir_pair[1][0], j + dir_pair[1][1]
                        p3_i, p3_j = i + dir_pair[2][0], j + dir_pair[2][1]
                        
                        if (0 <= p1_i < 9 and 0 <= p1_j < 9 and 
                            0 <= p2_i < 9 and 0 <= p2_j < 9 and 
                            reconstruction_mask[p1_i, p1_j] and 
                            reconstruction_mask[p2_i, p2_j]):
                            
                            p1 = reconstructed_grid[p1_i, p1_j]
                            p2 = reconstructed_grid[p2_i, p2_j]
                            using_original_points = valid_mask[p1_i, p1_j] and valid_mask[p2_i, p2_j]
                            
                            if (0 <= p3_i < 9 and 0 <= p3_j < 9 and 
                                reconstruction_mask[p3_i, p3_j]):
                                p3 = reconstructed_grid[p3_i, p3_j]
                                interpolated_point = extrapolate_point_with_ratio(p1, p2, p3)
                                
                                if using_original_points and valid_mask[p3_i, p3_j]:
                                    ratio_interpolations.append(interpolated_point)
                                    if debug_frame is not None:
                                        debug_lines.append((p1, p2, p3, "ratio"))
                                # else:
                                #     fallback_interpolations.append((interpolated_point, 
                                #         (point_weights[p1_i, p1_j] + point_weights[p2_i, p2_j] + point_weights[p3_i, p3_j]) / 3))
                                #     if debug_frame is not None:
                                #         debug_lines.append((p1, p2, p3, "fallback"))
                            else:
                                interpolated_point = extrapolate_point(p1, p2)
                                if using_original_points:
                                    simple_interpolations.append(interpolated_point)
                                    if debug_frame is not None:
                                        debug_lines.append((p1, p2, None, "simple"))
                                else:
                                    fallback_interpolations.append((interpolated_point,
                                        (point_weights[p1_i, p1_j] + point_weights[p2_i, p2_j]) / 2))
                                    if debug_frame is not None:
                                        debug_lines.append((p1, p2, None, "fallback"))
                    
                    if ratio_interpolations:
                        reconstructed_grid[i, j] = np.mean(ratio_interpolations, axis=0)
                        point_weights[i, j] = 0.8
                        color = (255, 255, 0)  # Cyan
                        debug_type = "ratio"
                    elif simple_interpolations:
                        reconstructed_grid[i, j] = np.mean(simple_interpolations, axis=0)
                        point_weights[i, j] = 0.6
                        color = (0, 255, 255)  # Jaune
                        debug_type = "simple"
                    elif fallback_interpolations:
                        points, weights = zip(*fallback_interpolations)
                        weights = np.array(weights) / np.sum(weights)
                        reconstructed_grid[i, j] = np.average(points, weights=weights, axis=0)
                        point_weights[i, j] = np.mean(weights) * 0.2
                        color = (0, 0, 255)  # Rouge
                        debug_type = "fallback"
                    else:
                        continue
                    
                    reconstruction_mask[i, j] = True
                    points_added = True
                    
                    # Debug visualization
                    if debug_frame is not None:
                        x, y = map(int, reconstructed_grid[i, j])
                        cv2.circle(debug_frame, (x, y), 4, color, -1)
                        
                        # Dessiner les lignes d'interpolation
                        for p1, p2, p3, interp_type in debug_lines:

                            if interp_type == debug_type:

                                pt1 = tuple(map(int, p1))
                                pt2 = tuple(map(int, p2))
                                pt_final = (x, y)


                                # Pause pour visualiser l'ajout de chaque ligne
                                # while True:
                                #     cv2.imshow('Debug Frame', debug_frame)
                                #     key = cv2.waitKey(1) & 0xFF
                                #     if key == ord('l'):  # Continuer sur appui de 'l'
                                #         break
                                #     elif key == ord('q'):  # Quitter sur appui de 'q'
                                #         cv2.destroyAllWindows()
                                #         exit()
                                
                                # Déterminer la couleur de la ligne en fonction des points
                                if using_original_points:  # Les deux points sont d'origine
                                    line_color = (255, 255, 0)  # Cyan
                                elif (0 <= p1_i < 9 and 0 <= p1_j < 9 and valid_mask[p1_i, p1_j]) or \
                                    (0 <= p2_i < 9 and 0 <= p2_j < 9 and valid_mask[p2_i, p2_j]):  
                                    line_color = (0, 255, 255)  # Jaune
                                else:  # Aucun point d'origine
                                    line_color = (0, 0, 255)    # Rouge
                                
                                cv2.line(debug_frame, pt1, pt2, line_color, 1)

                            # cv2.arrowedLine(debug_frame, pt1, pt_final, line_color, 2, tipLength=0.2)
                            # if p3 is not None and interp_type == "ratio":
                            #     pt3 = tuple(map(int, p3))
                            #     # Vérifier si p3 est dans les limites et est d'origine
                            #     if 0 <= p3_i < 9 and 0 <= p3_j < 9 and valid_mask[p3_i, p3_j]:
                            #         if using_original_points:  # Tous les points sont d'origine
                            #             line_color = (255, 255, 0)  # Cyan
                            #         else:  # Au moins un point n'est pas d'origine
                            #             line_color = (0, 255, 255)  # Jaune
                            #     else:
                            #         line_color = (0, 0, 255)  # Rouge
                            #     cv2.arrowedLine(debug_frame, pt3, pt_final, line_color, 2, tipLength=0.2)
    
    return reconstructed_grid

