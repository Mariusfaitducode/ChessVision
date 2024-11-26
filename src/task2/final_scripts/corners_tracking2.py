import cv2
import numpy as np
from utils import *

def estimate_corners_movement(prev_grid, prev_mask, curr_frame, prev_frame, debug=False):
    """
    Estime la position des coins en utilisant uniquement les points fiables de la frame précédente.
    
    Args:
        prev_grid: Grille précédente (9x9x2)
        prev_mask: Masque des points fiables de prev_grid (9x9 bool)
        curr_frame: Frame courante
        prev_frame: Frame précédente
        debug: Afficher la visualisation du tracking
    
    Returns:
        corners: Les 4 coins du plateau (4x2) ou None
        new_grid: Grille mise à jour avec les nouveaux points (9x9x2)
        new_mask: Nouveau masque des points fiables (9x9 bool)
    """
    if prev_frame is None or prev_grid is None:
        return None, None, None
    
    # Extraire uniquement les points fiables des bordures
    border_points, border_info, point_positions = extract_border_points(prev_grid, prev_mask)
    
    if len(border_points) < 4:
        return None, prev_grid, None
    
    # Calculer le flux optique
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    prev_points = border_points.reshape(-1, 1, 2).astype(np.float32)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    if not np.all(status == 1):
        return None, prev_grid, None
    
    # Valider les points trackés
    valid_mask, invalid_reasons = validate_tracked_points(prev_points, next_points, curr_frame, prev_frame, border_info)
    valid_points = next_points[valid_mask].reshape(-1, 2)
    valid_border_info = border_info[valid_mask]
    valid_positions = [pos for i, pos in enumerate(point_positions) if valid_mask[i]]
    
    
    
    # Créer la nouvelle grille et le nouveau masque
    new_grid = prev_grid.copy()
    new_mask = prev_mask.copy()
    new_mask[~prev_mask] = False
    
    # Mettre à jour la grille et le masque
    for (i, j), point, is_valid in zip(point_positions, next_points, valid_mask):
        if is_valid:
            new_grid[i, j] = point.flatten()
        else:
            new_mask[i, j] = False
        
    
    # Séparer les points par bordure
    top_points = valid_points[valid_border_info == 0]
    right_points = valid_points[valid_border_info == 1]
    bottom_points = valid_points[valid_border_info == 2]
    left_points = valid_points[valid_border_info == 3]
    
    if len(top_points) < 2 or len(right_points) < 2 or len(bottom_points) < 2 or len(left_points) < 2:
        return None, new_grid, new_mask
    
    # Ajuster les lignes
    lines = [
        cv2.fitLine(top_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01),
        cv2.fitLine(bottom_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01),
        cv2.fitLine(left_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01),
        cv2.fitLine(right_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    ]
    
    if debug:
        debug_frame = curr_frame.copy()
        # Dessiner les lignes plus fines
        draw_lines(debug_frame, lines, thickness=1)
    
    corners = find_intersections(*lines)
    if corners is None:
        return None, new_grid, new_mask
    

    # if debug:
    #     debug_frame = curr_frame.copy()
    #     colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        
    #     # Dessiner d'abord les lignes RANSAC pour chaque bordure
    #     for border in range(4):
    #         border_mask = (border_info == border) & valid_mask
    #         if np.sum(border_mask) >= 2:
    #             border_points = next_points[border_mask].reshape(-1, 2)
    #             line, _ = fit_line_ransac(border_points)
    #             if line is not None:
    #                 vx, vy, x, y = line.flatten()
    #                 pt1 = (int(x - vx*1000), int(y - vy*1000))
    #                 pt2 = (int(x + vx*1000), int(y + vy*1000))
    #                 cv2.line(debug_frame, pt1, pt2, colors[border], 1)
        
    #     # Dessiner les points et les flèches
    #     for prev_pt, next_pt, border_type, is_valid, reason in zip(prev_points, next_points, border_info, valid_mask, invalid_reasons):
    #         p1 = tuple(map(int, prev_pt[0]))
    #         p2 = tuple(map(int, next_pt[0]))
            
    #         if is_valid:

    #             print('VALIDDD')
    #             # Points valides
    #             cv2.circle(debug_frame, p1, 2, colors[border_type], -1)
    #             cv2.arrowedLine(debug_frame, p1, p2, colors[border_type], 1, tipLength=0.2)
    #             cv2.circle(debug_frame, p2, 3, colors[border_type], -1)
    #         else:
    #             # Points invalidés
    #             cv2.circle(debug_frame, p1, 2, (0,0,255), -1)  # Point de départ en rouge
    #             cv2.circle(debug_frame, p2, 3, (128,0,128), -1)  # Point d'arrivée en violet
    #             cv2.line(debug_frame, p1, p2, (0,0,255), 1, cv2.LINE_AA)  # Ligne rouge en pointillés
                
    #             # Afficher le numéro de raison aux deux extrémités
    #             cv2.putText(debug_frame, str(reason), p1, 
    #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #             cv2.putText(debug_frame, str(reason), p2, 
    #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,128), 1)
        
    #     cv2.imshow('Border Tracking', debug_frame)
    #     cv2.waitKey(1)
    
    if debug:
        # Dessiner les coins
        for corner in corners:
            cv2.circle(debug_frame, tuple(map(int, corner)), 5, (0, 0, 255), -1)

        
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]  # couleur par bordure
        


        for prev_pt, next_pt, border_type, is_valid, reason in zip(prev_points, next_points, border_info, valid_mask, invalid_reasons):
            p1 = tuple(map(int, prev_pt[0]))
            p2 = tuple(map(int, next_pt[0]))
            
            # Dessiner le point de départ
            cv2.circle(debug_frame, p1, 3, (0, 0, 0), -1)
            
            if is_valid:
                # Dessiner la flèche de mouvement
                cv2.arrowedLine(debug_frame, p1, p2, (255, 255, 0), 1, tipLength=0.2)
                # Dessiner le point d'arrivée
                cv2.circle(debug_frame, p2, 3, (255, 255, 0), -1)
            else:
                # Afficher la raison d'invalidation
                cv2.circle(debug_frame, p1, 2, (0,0,255), -1)  # Point de départ en rouge
                cv2.circle(debug_frame, p2, 3, (128,0,128), -1)  # Point d'arrivée en violet
                cv2.line(debug_frame, p1, p2, (0,0,255), 1, cv2.LINE_AA)  # Ligne rouge en pointillés
                
                # Afficher le numéro de raison aux deux extrémités
                # cv2.putText(debug_frame, str(reason), p1, 
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.putText(debug_frame, str(reason), p2, 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.imshow('Border Tracking', debug_frame)
        
        if debug:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Espace
                    break
                elif key == ord('q'):  # q
                    cv2.destroyAllWindows()
                    exit()

    
    return corners, new_grid, new_mask



def extract_border_points(grid, mask):
    """
    Extrait les points fiables des bordures de la grille.
    Retourne aussi les positions (i,j) de chaque point dans la grille.
    """
    points = []
    border_info = []
    point_positions = []
    
    # Top border
    for j in range(grid.shape[1]):
        if mask[0, j]:
            points.append(grid[0, j])
            border_info.append(0)
            point_positions.append((0, j))
    
    # Right border
    for i in range(1, grid.shape[0]-1):
        if mask[i, -1]:
            points.append(grid[i, -1])
            border_info.append(1)
            point_positions.append((i, grid.shape[1]-1))
    
    # Bottom border
    for j in range(grid.shape[1]-1, -1, -1):
        if mask[-1, j]:
            points.append(grid[-1, j])
            border_info.append(2)
            point_positions.append((grid.shape[0]-1, j))
    
    # Left border
    for i in range(grid.shape[0]-2, 0, -1):
        if mask[i, 0]:
            points.append(grid[i, 0])
            border_info.append(3)
            point_positions.append((i, 0))
    
    return np.array(points), np.array(border_info), point_positions




def validate_tracked_points(prev_points, next_points, curr_frame, prev_frame, border_info, max_movement_ratio=2):
    """
    Valide les points trackés et retourne un masque des points valides.
    
    Args:
        prev_points: Points de la frame précédente (Nx1x2)
        next_points: Points trackés dans la frame courante (Nx1x2)
        curr_frame: Frame courante
        prev_frame: Frame précédente
        max_movement_ratio: Ratio maximum de mouvement par rapport à la médiane
    
    Returns:
        valid_mask: Masque booléen (N) indiquant les points valides
        invalid_reasons: Liste des raisons d'invalidation (1, 2 ou 3) pour chaque point invalide
    """
    N = len(prev_points)
    valid_mask = np.ones(N, dtype=bool)
    invalid_reasons = np.zeros(N, dtype=int)  # 0 = valide, 1-3 = raison d'invalidation
    
    # 1. Validation par cohérence de mouvement
    # movement_vectors = next_points - prev_points
    # movements = np.linalg.norm(movement_vectors.reshape(-1, 2), axis=1)
    # median_movement = np.median(movements)
    
    # movement_mask = movements < (median_movement * max_movement_ratio)
    # valid_mask &= movement_mask
    # invalid_reasons[~movement_mask] = 1
    
    # 2. Validation par intensité locale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    window_size = 5
    intensity_threshold = 50
    
    for i, (valid, prev_pt, next_pt) in enumerate(zip(valid_mask, prev_points, next_points)):
        if valid:
            # Extraire les fenêtres autour du point dans les deux frames
            x, y = map(int, next_pt[0])
            prev_x, prev_y = map(int, prev_pt[0])
            
            if (0 <= x < curr_gray.shape[1] - window_size and 
                0 <= y < curr_gray.shape[0] - window_size and
                0 <= prev_x < prev_gray.shape[1] - window_size and 
                0 <= prev_y < prev_gray.shape[0] - window_size):
                
                curr_window = curr_gray[y:y+window_size, x:x+window_size]
                prev_window = prev_gray[prev_y:prev_y+window_size, prev_x:prev_x+window_size]
                
                if np.abs(np.mean(curr_window) - np.mean(prev_window)) > intensity_threshold:
                    valid_mask[i] = False
                    invalid_reasons[i] = 2
            else:
                valid_mask[i] = False
                invalid_reasons[i] = 2
    
    # 3. Validation par détection de coins Harris
    corners_harris = cv2.cornerHarris(curr_gray, blockSize=2, ksize=3, k=0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    
    search_radius = 10
    corner_threshold = 0.01 * corners_harris.max()
    
    for i, (valid, next_pt) in enumerate(zip(valid_mask, next_points)):
        if valid:
            x, y = map(int, next_pt[0])
            if 0 <= x < curr_gray.shape[1] and 0 <= y < curr_gray.shape[0]:
                y_min = max(0, y - search_radius)
                y_max = min(curr_gray.shape[0], y + search_radius)
                x_min = max(0, x - search_radius)
                x_max = min(curr_gray.shape[1], x + search_radius)
                
                region = corners_harris[y_min:y_max, x_min:x_max]
                if region.max() < corner_threshold:
                    valid_mask[i] = False
                    invalid_reasons[i] = 3
            else:
                valid_mask[i] = False
                invalid_reasons[i] = 3
    
    # 4. Validation par alignement des points sur chaque bordure
    for border in range(4):
        # Récupérer les points valides de cette bordure
        border_mask = (border_info == border) & valid_mask
        if np.sum(border_mask) < 2:
            continue
            
        border_points = next_points[border_mask].reshape(-1, 2)
        
        # Utiliser RANSAC pour trouver la meilleure ligne
        line, inliers = fit_line_ransac(border_points)
        
        if line is not None:
            # Marquer les outliers comme invalides
            current_idx = 0
            for i, is_border in enumerate(border_info == border):
                if is_border and valid_mask[i]:
                    if not inliers[current_idx]:
                        valid_mask[i] = False
                        invalid_reasons[i] = 4
                    current_idx += 1
    
    return valid_mask, invalid_reasons




def separate_border_points(points, points_per_border):
    """Sépare les points trackés en 4 bordures en les regroupant par position."""
    points = points.reshape(-1, 2)
    
    # Calculer le centre de tous les points
    center = np.mean(points, axis=0)
    
    # Séparer les points selon leur position par rapport au centre
    top_points = []
    bottom_points = []
    left_points = []
    right_points = []
    
    for point in points:
        x, y = point
        dx = x - center[0]
        dy = y - center[1]
        
        # Déterminer à quelle bordure appartient le point
        if abs(dy) > abs(dx):  # Point plus vertical qu'horizontal
            if dy < 0:  # Au-dessus du centre
                top_points.append(point)
            else:  # En-dessous du centre
                bottom_points.append(point)
        else:  # Point plus horizontal que vertical
            if dx < 0:  # À gauche du centre
                left_points.append(point)
            else:  # À droite du centre
                right_points.append(point)
    
    return (np.array(top_points), np.array(bottom_points), 
            np.array(left_points), np.array(right_points))

def fit_line_robust(points):
    """Ajuste une ligne de manière robuste en utilisant RANSAC."""
    if len(points) < 2:
        return None
    
    # Convertir en format attendu par cv2.fitLine
    points = points.reshape(-1, 1, 2).astype(np.float32)
    
    # Utiliser RANSAC pour un ajustement plus robuste
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
    return np.array([vx[0], vy[0], x[0], y[0]])

def find_intersections(top_line, bottom_line, left_line, right_line):
    """Trouve les intersections des lignes pour obtenir les 4 coins."""
    def line_intersection(line1, line2):
        vx1, vy1, x1, y1 = line1.flatten()
        vx2, vy2, x2, y2 = line2.flatten()
        
        # Résoudre le système d'équations pour trouver l'intersection
        A = np.array([[-vy1, vx1], [-vy2, vx2]])
        b = np.array([[-vy1*x1 + vx1*y1], [-vy2*x2 + vx2*y2]])
        
        try:
            x, y = np.linalg.solve(A, b)
            return np.array([x[0], y[0]])
        except np.linalg.LinAlgError:
            return None
    
    # Trouver les 4 intersections
    top_left = line_intersection(top_line, left_line)
    top_right = line_intersection(top_line, right_line)
    bottom_left = line_intersection(bottom_line, left_line)
    bottom_right = line_intersection(bottom_line, right_line)
    
    # Vérifier si toutes les intersections ont été trouvées
    if any(corner is None for corner in [top_left, top_right, bottom_left, bottom_right]):
        return None
    
    return np.array([top_left, bottom_left, bottom_right, top_right])

def draw_lines(frame, lines, thickness=1, length=1000):
    """Dessine les lignes de bordure sur l'image de debug."""
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    
    for line, color in zip(lines, colors):
        vx, vy, x, y = line.flatten()
        pt1 = (int(x - vx*length), int(y - vy*length))
        pt2 = (int(x + vx*length), int(y + vy*length))
        cv2.line(frame, pt1, pt2, color, thickness)

def fit_line_ransac(points, max_deviation=10, min_inliers_ratio=0.5):
    """
    Trouve la meilleure ligne en testant toutes les combinaisons de points possibles.
    
    Args:
        points: Points à ajuster (Nx2)
        max_deviation: Distance maximum acceptée à la ligne (en pixels)
        min_inliers_ratio: Ratio minimum d'inliers requis
    """
    best_inliers = np.zeros(len(points), dtype=bool)
    best_line = None
    N = len(points)
    
    if N < 2:
        return None, best_inliers
    
    best_inliers_count = 0
    
    # Tester toutes les paires de points possibles
    for i in range(N-1):
        for j in range(i+1, N):
            p1, p2 = points[i], points[j]
            
            # Calculer la ligne passant par ces points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm < 1e-10:
                continue
            
            vx, vy = dx/norm, dy/norm
            x, y = p1
            
            # Calculer la distance de tous les points à cette ligne
            distances = np.abs(vy*(points[:,0]-x) - vx*(points[:,1]-y))
            
            # Compter les inliers
            inliers = distances < max_deviation
            inliers_count = np.sum(inliers)
            
            if inliers_count > best_inliers_count:
                best_inliers = inliers
                best_line = (vx, vy, x, y)
                best_inliers_count = inliers_count
    
    # Vérifier qu'on a suffisamment d'inliers
    if best_inliers_count < N * min_inliers_ratio:
        return None, np.zeros(N, dtype=bool)
    
    if best_line is not None and best_inliers_count >= 2:
        # Réajuster la ligne finale avec tous les inliers
        best_line = cv2.fitLine(points[best_inliers].astype(np.float32), 
                              cv2.DIST_L2, 0, 0.01, 0.01)
    
    return best_line, best_inliers
