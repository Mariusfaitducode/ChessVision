import trimesh
import numpy as np
import cv2



def load_chess_piece(piece_name):
    # Charger le mesh avec les matériaux
    mesh = trimesh.load(f'assets/3d_models/{piece_name}.obj')
    
    mesh.vertices -= mesh.vertices.mean(axis=0)
    
    # Augmenter l'échelle
    scale_factor = 35
    mesh.vertices *= scale_factor
    
    # Rotation de 90 degrés autour de l'axe X pour redresser la pièce
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    mesh.vertices = mesh.vertices.dot(rotation_matrix)


    # Rotation de 180 degrés autour de l'axe Z
    rotation_matrix_z = np.array([
        [-1, 0, 0],   # cos(180°)=-1, sin(180°)=0
        [0, -1, 0],   # Inverse les directions X et Y
        [0, 0, 1]     # Z reste inchangé
    ])
    mesh.vertices = mesh.vertices.dot(rotation_matrix_z)

    
    # Ajuster la hauteur pour que la pièce repose sur le plateau
    min_z = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= min_z
    
    # Utiliser les valeurs du MTL pour la couleur de base
    # Kd du MTL est 0.8, 0.8, 0.8 donc on utilise ces valeurs * 255

    if piece_name.startswith('white'):
        base_color = np.array([0.8 * 255, 0.8 * 255, 0.8 * 255])
    else:
        base_color = np.array([0.2 * 255, 0.2 * 255, 0.2 * 255])
    
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces), base_color




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
            
            if game_actualization['piece_certainty'][(i, j)] != {}:

                certainties = game_actualization['piece_certainty'][(i, j)]

                CERTAINTY_THRESHOLD = 0.7
                for piece, prob in certainties.items():
                    if prob > CERTAINTY_THRESHOLD:
                        
                        display_chess_piece(frame, params, piece, i, j)

    return frame




def display_chess_piece(frame, params, piece_name, case_i, case_j):
    rvec = params['rvec']
    tvec = params['tvec']
    camera_matrix = params['camera_matrix']
    dist_coeffs = params['dist_coeffs']

    vertices, faces, base_color = load_chess_piece(piece_name)

    # Déplacer la pièce à la position de la case
    # On considère que le plateau fait 8x8 cases et est centré à l'origine
    offset_x = (case_j) * 75 + 40  # -3.5 pour centrer (0-7 -> -3.5 à 3.5)
    offset_y = (case_i) * 75 + 40
    
    transformed_vertices = vertices.copy()
    transformed_vertices[:, 0] += offset_x
    transformed_vertices[:, 1] += offset_y
            
    # Direction de la lumière
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    projected_points, _ = cv2.projectPoints(
        transformed_vertices,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    
    projected_points = projected_points.reshape(-1, 2).astype(np.int32)
    
    # Dessiner les faces avec ombrage
    for face in faces:
        pts = np.array([
            projected_points[face[0]],
            projected_points[face[1]],
            projected_points[face[2]]
        ])
        
        # Calculer la normale et l'éclairage
        v1 = transformed_vertices[face[1]] - transformed_vertices[face[0]]
        v2 = transformed_vertices[face[2]] - transformed_vertices[face[0]]
        face_normal = np.cross(v1, v2)
        
        norm = np.linalg.norm(face_normal)
        if norm > 1e-10:
            face_normal = face_normal / norm
            # Combiner l'éclairage ambiant (Ka) et diffus (Kd) du MTL
            ambient = 0.6  # Ka du MTL est 1.0, mais on le réduit pour l'effet
            diffuse = np.abs(np.dot(face_normal, light_dir))
            intensity = ambient + (1 - ambient) * diffuse
            
            # Appliquer l'intensité à la couleur de base
            color_value = base_color * intensity
            color_value = np.clip(color_value, 0, 255)
            color = tuple(map(int, color_value))
            
            cv2.fillPoly(frame, [pts], color=color)
            
            # Ajouter un effet spéculaire subtil (Ks du MTL)
            spec_color = tuple(map(int, np.minimum(color_value + 30, 255)))
            cv2.polylines(frame, [pts], isClosed=True, color=spec_color, thickness=1)

    return frame