import numpy as np
import matplotlib.pyplot as plt
import cv2



def detect_if_case_is_occupied(edges_frame, blurred_frame, inner_top, inner_left, inner_bottom, inner_right):
    
    # Extract square region for edge detection (inner area only)
    square_edges = edges_frame[inner_top:inner_bottom, inner_left:inner_right]
    
    # Count edge pixels in inner area
    edge_count = np.count_nonzero(square_edges)
    
    # Calculate edge pixel percentage
    # Note: using inner area for calculation
    inner_area = (inner_bottom - inner_top) * (inner_right - inner_left)
    edge_percentage = (edge_count / inner_area) * 100
    
    

    square_blurred = blurred_frame[inner_top:inner_bottom, inner_left:inner_right]

    pixel_variance = np.var(square_blurred)
    

    # Threshold to determine if square is occupied
    edge_threshold = 1  # Adjust this threshold as needed
    # A square is considered occupied if it contains enough edges
    is_occupied = bool(edge_percentage > 1 or pixel_variance > 50)

    return is_occupied, edge_percentage, pixel_variance




def get_piece_roi(piece_region):
    # Appliquer Canny pour détecter les contours
    edges = cv2.Canny(piece_region, 50, 150)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Trouver le plus grand contour (qui devrait être la pièce)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Obtenir le rectangle englobant
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Ajouter une petite marge autour de la pièce (20% de la taille)
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    # Calculer les nouvelles coordonnées avec la marge
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(piece_region.shape[1], x + w + margin_x)
    y2 = min(piece_region.shape[0], y + h + margin_y)
    
    return (x1, y1, x2, y2)




def detect_piece_color(piece_region, is_dark_square, debug=False):

    # Prétraitement de l'image
    blurred = cv2.GaussianBlur(piece_region, (5, 5), 0)
    
    # Obtenir la ROI centrée sur la pièce
    roi_coords = get_piece_roi(blurred)
    if roi_coords is None:
        return None
    
    x1, y1, x2, y2 = roi_coords
    piece_roi = piece_region[y1:y2, x1:x2]


    # Calculer l'histogramme
    hist = cv2.calcHist([piece_roi], [0], None, [256], [0, 256])
    
    # Lisser l'histogramme pour réduire le bruit
    hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)

    # Trouver les pics significatifs
    peaks = []
    min_peak_height = np.max(hist_smooth) * 0.01  # 1% du maximum pour détecter les petits pics

    if is_dark_square:
        min_peak_distance = 15  # Distance minimale entre deux pics
    else:
        min_peak_distance = 8  # Distance minimale entre deux pics
    
    # Trouver les pics significatifs
    peaks = []
    for i in range(1, 255):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            # Vérifier si le pic est assez haut
            if hist_smooth[i] > min_peak_height:
                # Vérifier si le pic est assez loin des pics existants
                is_far_enough = True
                for existing_peak, _ in peaks:
                    if abs(existing_peak - i) < min_peak_distance:
                        is_far_enough = False
                        # Si le nouveau pic est plus haut, remplacer l'ancien
                        if hist_smooth[i] > hist_smooth[existing_peak]:
                            peaks.remove((existing_peak, hist_smooth[existing_peak][0]))
                            is_far_enough = True
                            break
                
                if is_far_enough:
                    peaks.append((i, hist_smooth[i][0]))
    
    # Trier les pics par hauteur
    peaks.sort(key=lambda x: x[1], reverse=True)

    if debug:
        plt.figure(figsize=(15, 5))
        
        # Image originale avec ROI
        plt.subplot(131)
        plt.imshow(piece_region, cmap='gray')
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
        plt.title('Image originale avec ROI')
        
        # ROI extraite
        plt.subplot(132)
        plt.imshow(piece_roi, cmap='gray')
        plt.title('ROI de la pièce')
        
        # Histogramme
        plt.subplot(133)
        plt.plot(hist_smooth, 'b-', label='Histogramme lissé')
        plt.plot(hist, 'r:', alpha=0.5, label='Histogramme brut')
        
        for i, (pos, height) in enumerate(peaks):
            plt.plot(pos, height, 'go', markersize=10, label=f'Pic {i+1} (val={pos})')
            plt.text(pos, height, f'  {pos}', verticalalignment='bottom')
        
        plt.title(f"Histogramme de la ROI")
        plt.legend()
        plt.grid(True)
        plt.show()
    


    
    if len(peaks) < 2:
        return None, None  # Pas assez de pics pour une détection fiable
    
    # Le plus grand pic correspond au fond
    background_peak = peaks[0][0]
    # Le deuxième pic le plus important correspond à la pièce
    piece_peak = peaks[1][0]
            
    if abs(piece_peak - 130) < 40:
        return 'white', piece_peak
    
    # Pièce blanche (pic autour de 210)
    elif abs(piece_peak - 210) < 30:
        return 'black', piece_peak
    
    return None, piece_peak
