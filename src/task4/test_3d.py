import cv2
import trimesh
import numpy as np

# Charger le modèle 3D
mesh = trimesh.load(r'C:\Users\mariu\Desktop\CodeProjects\Computer Vision\ChessVision\src\task4\white_king.obj')  # Remplacez par le chemin vers votre modèle

# Centrer et mettre à l'échelle le modèle
mesh.vertices -= mesh.vertices.mean(axis=0)  # Centrer le modèle
scale_factor = 100.0  # Augmenté pour une meilleure visibilité
mesh.vertices *= scale_factor

# Obtenir les sommets 3D et les faces
vertices = np.array(mesh.vertices, dtype=np.float32)  # Assurez-vous que les sommets sont en float32
faces = np.array(mesh.faces)  # Les indices des sommets n'ont pas besoin d'être convertis

# Matrice intrinsèque de la caméra (float32)
camera_matrix = np.array([
    [800, 0, 320],  # Focale en x, et centre optique x
    [0, 800, 240],  # Focale en y, et centre optique y
    [0, 0, 1]       # Normalisation
], dtype=np.float32)

# Ajuster la position et la rotation pour une meilleure vue
rotation_vector = np.array([0, 0, 3.14], dtype=np.float32)  # Rotation modifiée
translation_vector = np.array([0, 0, 800], dtype=np.float32)  # Distance augmentée

# Projeter les sommets 3D dans le plan 2D
projected_points, _ = cv2.projectPoints(
    vertices, 
    rotation_vector, 
    translation_vector, 
    camera_matrix, 
    None  # Pas de distorsion
)

# Convertir les points projetés pour OpenCV
projected_points = projected_points.squeeze().astype(int)

# Créer une image avec un fond noir pour un meilleur contraste
image = np.zeros((480, 640, 3), dtype=np.uint8)

# Améliorer le rendu des faces
for face in faces:
    pts = np.array([projected_points[face[0]], 
                    projected_points[face[1]], 
                    projected_points[face[2]]], dtype=np.int32)
    cv2.fillPoly(image, [pts], color=(180, 180, 180))  # Couleur plus claire
    cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 255), thickness=1)  # Contours blancs

# Afficher l'image résultante
cv2.imshow("3D Model Projection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
