import cv2
import numpy as np


def estimate_corners_movement(prev_corners, curr_frame, prev_frame, debug=False):
    """Estime le mouvement des coins entre deux frames en utilisant le flux optique"""
    if prev_frame is None or prev_corners is None:
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Convertir la liste en array numpy
    prev_corners_array = np.array(prev_corners, dtype=np.float32)
    # Reshape pour le format attendu par calcOpticalFlowPyrLK: (N, 1, 2)
    prev_corners_array = prev_corners_array.reshape(-1, 1, 2)
    
    # Calculer le flux optique
    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_corners_array, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Visualisation
    if debug:
        vis_img = curr_frame.copy()
        
        # Dessiner les coins précédents en rouge
        for corner in prev_corners_array:
            x, y = corner.ravel()
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Dessiner les nouveaux coins en vert et les flèches de mouvement
        for i, (new, s) in enumerate(zip(next_corners, status)):
            if s:  # Si le point a été bien suivi
                x, y = new.ravel()
                cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # Dessiner une flèche du point précédent au nouveau point
                prev_x, prev_y = prev_corners_array[i].ravel()
                cv2.arrowedLine(vis_img, 
                              (int(prev_x), int(prev_y)), 
                              (int(x), int(y)),
                              (255, 0, 0), 2)  # Flèche bleue
        
        # Afficher l'image
        cv2.imshow('Optical Flow Tracking', vis_img)
        cv2.waitKey(1)
    
    # Filtrer les coins valides et retourner au format liste
    valid_corners = next_corners[status == 1]
    return valid_corners.reshape(-1, 2).tolist()  # Convertir en liste pour maintenir le format

def validate_and_refine_corner(img, estimated_pos, search_radius=20):
    """Valide et affine la position d'un coin estimé en utilisant Harris"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners_harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    
    x, y = int(estimated_pos[0]), int(estimated_pos[1])
    
    # Vérifier si le point est dans l'image
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        return None
    
    # Définir la région de recherche
    y_min = max(0, y - search_radius)
    y_max = min(gray.shape[0], y + search_radius)
    x_min = max(0, x - search_radius)
    x_max = min(gray.shape[1], x + search_radius)
    
    # Extraire la région d'intérêt
    region = corners_harris[y_min:y_max, x_min:x_max]
    
    # Vérifier si un coin fort existe dans la région
    if region.max() > 0.01:  # Seuil ajustable
        y_local, x_local = np.unravel_index(np.argmax(region), region.shape)
        return np.array([x_min + x_local, y_min + y_local])
    return None

def detect_corners(frame, prev_frame=None, prev_corners=None):
    """Détecte les coins du plateau d'échecs avec fallback sur le suivi"""
    # Essayer d'abord la détection classique
    corners = cv2.findChessboardCorners(frame, (7, 7))
    
    if corners[0]:  # Si la détection réussit
        return corners[1]
    
    # Si la détection échoue mais qu'on a des données précédentes
    if prev_corners is not None and prev_frame is not None:
        # Estimer le mouvement des coins
        estimated_corners = estimate_corners_movement(prev_corners, frame, prev_frame)
        if estimated_corners is None:
            return None
        
        # Valider et affiner chaque coin estimé
        refined_corners = []
        for corner in estimated_corners:
            refined_pos = validate_and_refine_corner(frame, corner)
            if refined_pos is not None:
                refined_corners.append(refined_pos)
        
        # Vérifier si on a assez de coins valides (par exemple, au moins 60%)
        if len(refined_corners) >= 0.6 * len(prev_corners):
            return np.array(refined_corners)
    
    return None