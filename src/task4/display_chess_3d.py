import trimesh
import numpy as np
import cv2



def load_chess_piece():
    mesh = trimesh.load('src/task4/white_king.obj')
    mesh.vertices -= mesh.vertices.mean(axis=0)
    
    # Augmenter l'échelle
    scale_factor = 25
    mesh.vertices *= scale_factor
    
    # Rotation de 90 degrés autour de l'axe X pour redresser la pièce
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    mesh.vertices = mesh.vertices.dot(rotation_matrix)
    
    # Ajuster la hauteur pour que la pièce repose sur le plateau
    min_z = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= min_z
    
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces)




def display_chess_game_3d(frame, params, game_actualization):

    # Chessboard dimensions
    height, width = frame.shape[:2]
    square_h = height // 8
    square_w = width // 8

    if game_actualization == {}:
        return frame

    # Draw rectangles and add text
    for i in range(8):
        for j in range(8):
            
            if game_actualization['piece_certainty'][(j, i)] != {}:

                certainties = game_actualization['piece_certainty'][(j, i)]

                

                

                CERTAINTY_THRESHOLD = 0.7
                for piece, prob in certainties.items():
                    if prob > CERTAINTY_THRESHOLD:
                        
                        display_chess_piece(frame, params, i, j)

    return frame




def display_chess_piece(frame, params, case_i, case_j):
    """
    Affiche une pièce d'échecs 3D sur une case spécifique du plateau
    params: dictionnaire contenant rvec, tvec, camera_matrix, dist_coeffs
    case_i, case_j: indices de la case sur le plateau (0-7)
    """
    rvec = params['rvec']
    tvec = params['tvec']
    camera_matrix = params['camera_matrix']
    dist_coeffs = params['dist_coeffs']

    vertices, faces = load_chess_piece()

    # Déplacer la pièce à la position de la case
    # On considère que le plateau fait 8x8 cases et est centré à l'origine
    offset_x = (case_j) * 75 + 40  # -3.5 pour centrer (0-7 -> -3.5 à 3.5)
    offset_y = (case_i) * 75 + 40
    
    # Appliquer l'offset aux vertices
    transformed_vertices = vertices.copy()
    transformed_vertices[:, 0] += offset_x
    transformed_vertices[:, 1] += offset_y
            
    # Projeter les points
    projected_points, _ = cv2.projectPoints(
        transformed_vertices,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    
    projected_points = projected_points.reshape(-1, 2).astype(np.int32)
    
    # Dessiner le modèle
    for face in faces:
        pts = np.array([
            projected_points[face[0]],
            projected_points[face[1]],
            projected_points[face[2]]
        ])
        
        cv2.fillPoly(frame, [pts], color=(200, 200, 200))
        cv2.polylines(frame, [pts], isClosed=True, color=(100, 100, 100), thickness=1)

    return frame