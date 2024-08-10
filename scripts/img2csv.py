import cv2
import pandas as pd
import numpy as np

def canny_edge_detection(image_path, output_csv_path):
    # Load the image
    img = cv2.imread(image_path)

    # Blur the image for better edge detection
    img = cv2.GaussianBlur(img,(5,5), sigmaX=0, sigmaY=0) 
    
    # Apply edge detection
    edges = cv2.Canny(img, 50, 60)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Prepare data for CSV
    data = []
    polyline_index = 0
    
    for contour in contours:
        # Convert contour to polyline
        polyline = contour[:, 0, :]  # Remove redundant dimension
        
        for point in polyline:
            x, y = point
            data.append([polyline_index, 0, x, y])
        
        polyline_index += 1
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['polyline_index', 'unused_col', 'x', 'y'])
    df.to_csv(output_csv_path, index=False)

def sobel_edge_detection(image_path, output_csv_path):
    img = cv2.imread(image_path)  # Ensure grayscale for edge detection
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0) 
    
    # Apply Sobel operators
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize and convert to uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)
    
    # Apply thresholding to get binary edge map
    _, binary_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Prepare data for CSV
    data = []
    polyline_index = 0
    
    for contour in contours:
        # Convert contour to polyline
        polyline = contour[:, 0, :]  # Remove redundant dimension
        
        for point in polyline:
            x, y = point
            data.append([polyline_index, 0, x, y])
        
        polyline_index += 1
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['polyline_index', 'unused_col', 'x', 'y'])
    df.to_csv(output_csv_path, index=False)

# Example usage
canny_edge_detection(r'misc-outputs\frag2.png', 'polylines.csv')
