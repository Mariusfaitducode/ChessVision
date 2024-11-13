import cv2
import numpy as np
import os

def analyze_chess_board(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection de contours avec Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dimensions de l'échiquier (8x8)
    rows, cols = 8, 8
    
    # Obtenir les dimensions de l'image
    height, width = gray.shape
    
    # Calculer la taille de chaque case
    square_h = height // rows
    square_w = width // cols
    
    # Dictionnaire pour stocker les résultats
    occupied_squares = {}
    square_stats = {}
    
    # Définir la marge (en pourcentage de la taille de la case)
    margin_percent = 0.15  # 15% de marge
    
    # Parcourir chaque case
    for i in range(rows):
        for j in range(cols):
            # Coordonnées de la case complète
            top = i * square_h
            left = j * square_w
            bottom = (i + 1) * square_h
            right = (j + 1) * square_w
            
            # Calculer les marges en pixels
            margin_h = int(square_h * margin_percent)
            margin_w = int(square_w * margin_percent)
            
            # Coordonnées de la zone d'analyse (avec marges)
            inner_top = top + margin_h
            inner_left = left + margin_w
            inner_bottom = bottom - margin_h
            inner_right = right - margin_w
            
            # Extraire la région de la case pour les contours (zone intérieure uniquement)
            square_edges = edges[inner_top:inner_bottom, inner_left:inner_right]
            
            # Compter les pixels de contours dans la zone intérieure
            edge_count = np.count_nonzero(square_edges)
            
            # Calculer le pourcentage de pixels de contours
            # Note: on utilise la surface de la zone intérieure pour le calcul
            inner_area = (inner_bottom - inner_top) * (inner_right - inner_left)
            edge_percentage = (edge_count / inner_area) * 100
            
            # Seuil pour déterminer si une case est occupée
            edge_threshold = 1  # Ajustez ce seuil selon vos besoins
            
            # Une case est considérée comme occupée si elle contient suffisamment de contours
            is_occupied = edge_percentage > edge_threshold
            
            # Stocker les résultats
            square_name = f"{chr(65+j)}{8-i}"
            occupied_squares[square_name] = is_occupied
            square_stats[square_name] = {
                'edge_percentage': edge_percentage
            }
            
            # Pour la visualisation, dessiner aussi la zone intérieure analysée
            # if current_view == 'edges':
            #     cv2.rectangle(img_display, 
            #                 (inner_left, inner_top), 
            #                 (inner_right, inner_bottom), 
            #                 (128, 128, 128), 1)  # Rectangle gris pour montrer la zone analysée
    
    return occupied_squares, img, {
        'gray': gray,
        'blurred': blurred,
        'edges': edges
    }, square_stats

def analyze_all_images(folder_path):
    results = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            squares, img, filtered_images, stats = analyze_chess_board(image_path)
            results[filename] = {
                'squares': squares,
                'image': img,
                'filtered': filtered_images,
                'stats': stats
            }
    
    # Visualisation interactive
    image_names = list(results.keys())
    current_idx = 0
    current_view = 'edges'  # 'original', 'gray', 'blurred', 'edges'
    
    while True:
        current_image = image_names[current_idx]
        result = results[current_image]
        
        # Sélectionner l'image à afficher selon le mode
        if current_view == 'original':
            img_display = result['image'].copy()
        else:
            img_display = cv2.cvtColor(result['filtered'][current_view], cv2.COLOR_GRAY2BGR)
        
        # Dimensions de l'échiquier
        height, width = img_display.shape[:2]
        square_h = height // 8
        square_w = width // 8
        
        # Dessiner les rectangles et ajouter le texte
        for i in range(8):
            for j in range(8):
                top = i * square_h
                left = j * square_w
                bottom = (i + 1) * square_h
                right = (j + 1) * square_w
                
                square_name = f"{chr(65+j)}{8-i}"
                is_occupied = result['squares'][square_name]
                stats = result['stats'][square_name]
                
                # Vert pour les cases occupées, rouge pour les cases vides
                color = (0, 255, 0) if is_occupied else (0, 0, 255)
                cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
                
                # Ajouter le pourcentage de contours
                text = f"edges:{stats['edge_percentage']:.1f}%"
                cv2.putText(img_display, text, (left + 5, top + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Afficher l'image
        cv2.imshow('Chess Analysis', img_display)
        print(f"\nAnalyse de {current_image} - Mode: {current_view}")
        
        # Gestion des touches
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('d'), 83]:  # Droite
            current_idx = (current_idx + 1) % len(image_names)
        elif key in [ord('a'), 81]:  # Gauche
            current_idx = (current_idx - 1) % len(image_names)
        elif key == ord('v'):  # Changer de vue
            views = ['original', 'gray', 'blurred', 'edges']
            current_view = views[(views.index(current_view) + 1) % len(views)]
    
    cv2.destroyAllWindows()
    return results

if __name__ == "__main__":
    folder_path = "images_results/warped_images"
    results = analyze_all_images(folder_path)
