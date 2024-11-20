import cv2
import numpy as np
import os

def analyze_chess_board(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Chessboard dimensions (8x8)
    rows, cols = 8, 8
    
    # Get image dimensions
    height, width = gray.shape
    
    # Calculate square size
    square_h = height // rows
    square_w = width // cols
    
    # Dictionary to store results
    occupied_squares = {}
    square_stats = {}
    
    # Define margin (as percentage of square size)
    margin_percent = 0.15  # 15% margin
    
    # Process each square
    for i in range(rows):
        for j in range(cols):
            # Full square coordinates
            top = i * square_h
            left = j * square_w
            bottom = (i + 1) * square_h
            right = (j + 1) * square_w
            
            # Calculate margins in pixels
            margin_h = int(square_h * margin_percent)
            margin_w = int(square_w * margin_percent)
            
            # Analysis zone coordinates (with margins)
            inner_top = top + margin_h
            inner_left = left + margin_w
            inner_bottom = bottom - margin_h
            inner_right = right - margin_w
            
            # Extract square region for edge detection (inner area only)
            square_edges = edges[inner_top:inner_bottom, inner_left:inner_right]
            
            # Count edge pixels in inner area
            edge_count = np.count_nonzero(square_edges)
            
            # Calculate edge pixel percentage
            # Note: using inner area for calculation
            inner_area = (inner_bottom - inner_top) * (inner_right - inner_left)
            edge_percentage = (edge_count / inner_area) * 100
            
            # Threshold to determine if square is occupied
            edge_threshold = 1  # Adjust this threshold as needed
            
            # A square is considered occupied if it contains enough edges
            is_occupied = edge_percentage > edge_threshold
            
            # Store results (chess notation: A1, B1, etc.)
            square_name = f"{chr(65+j)}{8-i}"
            occupied_squares[square_name] = is_occupied
            square_stats[square_name] = {
                'edge_percentage': edge_percentage
            }
            
            # For visualization, also draw the analyzed inner area
            
            cv2.rectangle(img, 
                        (inner_left, inner_top), 
                        (inner_right, inner_bottom), 
                        (128, 128, 128), 1)  # Gray rectangle to show analyzed area
    
    return occupied_squares, img, {
        'gray': gray,
        'blurred': blurred,
        'edges': edges
    }, square_stats

def analyze_all_images(folder_path):
    results = {}
    
    # Process all images in folder
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
    
    # Interactive visualization
    image_names = list(results.keys())
    current_idx = 0
    current_view = 'edges'  # 'original', 'gray', 'blurred', 'edges'
    
    while True:
        current_image = image_names[current_idx]
        result = results[current_image]
        
        # Select image to display based on current view
        if current_view == 'original':
            img_display = result['image'].copy()
        else:
            img_display = cv2.cvtColor(result['filtered'][current_view], cv2.COLOR_GRAY2BGR)
        
        # Chessboard dimensions
        height, width = img_display.shape[:2]
        square_h = height // 8
        square_w = width // 8
        
        # Draw rectangles and add text
        for i in range(8):
            for j in range(8):
                top = i * square_h
                left = j * square_w
                bottom = (i + 1) * square_h
                right = (j + 1) * square_w
                
                square_name = f"{chr(65+j)}{8-i}"
                is_occupied = result['squares'][square_name]
                stats = result['stats'][square_name]
                
                # Green for occupied squares, red for empty ones
                color = (0, 255, 0) if is_occupied else (0, 0, 255)
                cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
                
                # Add edge percentage text
                text = f"edges:{stats['edge_percentage']:.1f}%"
                cv2.putText(img_display, text, (left + 5, top + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display image
        cv2.imshow('Chess Analysis', img_display)
        print(f"\nAnalyzing {current_image} - Mode: {current_view}")
        
        # Key handling
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key in [ord('d'), 83]:  # Right arrow
            current_idx = (current_idx + 1) % len(image_names)
        elif key in [ord('s'), 81]:  # Left arrow
            current_idx = (current_idx - 1) % len(image_names)
        elif key == ord('v'):  # Change view
            views = ['original', 'gray', 'blurred', 'edges']
            current_view = views[(views.index(current_view) + 1) % len(views)]
    
    cv2.destroyAllWindows()
    return results

if __name__ == "__main__":
    folder_path = "images_results/warped_images"
    results = analyze_all_images(folder_path)
