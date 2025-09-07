import cv2
import numpy as np

def is_rail_line(binary_mask, threshold=0.6):
    """
    Simple function to check if SAM binary mask represents a rail line
    
    Args:
        binary_mask: (256, 256) numpy array with 0s and 1s
        threshold: Confidence threshold (0-1)
    
    Returns:
        bool: True if mask likely represents a line
    """
    
    mask = (binary_mask * 255).astype(np.uint8)
    
    if np.sum(mask > 0) < 10:  # Too few pixels
        return False
    
    score = 0
    max_score = 5
    
    # 1. Hough Line Detection
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=30, 
                           minLineLength=20, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        score += 1
    
    # 2. Aspect Ratio Test
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        width, height = rect[1]
        
        if min(width, height) > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3:  # Elongated shape
                score += 1
    
    # 3. PCA Linearity
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 5:
        points = np.column_stack([x_coords, y_coords])
        centered = points - np.mean(points, axis=0)
        
        try:
            eigenvals = np.linalg.eigvals(np.cov(centered.T))
            eigenvals = np.sort(eigenvals)[::-1]
            
            if eigenvals[1] > 0:
                linearity = eigenvals[0] / eigenvals[1]
                if linearity > 8:  # High linearity
                    score += 2
        except:
            pass
    
    # 4. Skeleton Analysis
    try:
        skeleton = cv2.ximgproc.thinning(mask)
        skeleton_pixels = np.sum(skeleton > 0)
        original_pixels = np.sum(mask > 0)
        
        if original_pixels > 0:
            skeleton_ratio = skeleton_pixels / original_pixels
            if skeleton_ratio < 0.4:  # Thin structure
                score += 1
    except:
        pass
    
    confidence = score / max_score
    return confidence >= threshold

def get_line_direction(binary_mask):
    """
    Get the primary direction of the line in the mask

    is:

    0° → line pointing to the right (east).

    90° → line pointing down (south).

    180° → line pointing left (west).

    That means:

    Angles increase clockwise in image space.

    The return is in [0,180] (because you normalize negative to positive and lines are bidirectional anyway).
    
    Returns:
        angle in degrees (0-180), or None if no clear line
    """
    
    mask = (binary_mask * 255).astype(np.uint8)
    
    # Try Hough lines first
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=20, 
                           minLineLength=15, maxLineGap=8)
    
    if lines is not None and len(lines) > 0:
        # Get angle of longest line
        longest_line = max(lines, key=lambda x: 
                          np.sqrt((x[0][2]-x[0][0])**2 + (x[0][3]-x[0][1])**2))
        
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Normalize to 0-180
        if angle < 0:
            angle += 180
            
        return angle
    
    # Fallback to PCA
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 3:
        points = np.column_stack([x_coords, y_coords])
        centered = points - np.mean(points, axis=0)
        
        try:
            eigenvals, eigenvecs = np.linalg.eig(np.cov(centered.T))
            principal_vec = eigenvecs[:, np.argmax(eigenvals)]
            angle = np.arctan2(principal_vec[1], principal_vec[0]) * 180 / np.pi
            
            if angle < 0:
                angle += 180
                
            return angle
        except:
            pass
    
    return None

# # Usage examples:
# def test_rail_detection():
#     """Test the rail detection functions"""
    
#     # Create test cases
#     test_cases = {
#         'horizontal_line': create_horizontal_line(),
#         'diagonal_line': create_diagonal_line(), 
#         'curved_line': create_curved_line(),
#         'blob': create_blob(),
#         'noise': create_noise()
#     }
    
#     for name, mask in test_cases.items():
#         is_line = is_rail_line(mask)
#         direction = get_line_direction(mask)
        
#         print(f"{name}: Line={is_line}, Direction={direction:.1f}° if direction else 'None'")

# def create_horizontal_line():
#     mask = np.zeros((256, 256))
#     mask[120:136, 50:200] = 1  # Thick horizontal line
#     return mask

# def create_diagonal_line():
#     mask = np.zeros((256, 256))
#     cv2.line(mask, (50, 50), (200, 200), 1, 8)
#     return mask

# def create_curved_line():
#     mask = np.zeros((256, 256))
#     # Create curved path
#     for i in range(50, 200):
#         y = int(128 + 30 * np.sin((i-50) * 0.05))
#         mask[max(0, y-3):min(256, y+4), i] = 1
#     return mask

# def create_blob():
#     mask = np.zeros((256, 256))
#     cv2.circle(mask, (128, 128), 40, 1, -1)  # Filled circle
#     return mask

# def create_noise():
#     mask = np.zeros((256, 256))
#     noise_points = np.random.randint(0, 256, (100, 2))
#     for point in noise_points:
#         mask[point[1], point[0]] = 1
#     return mask

# test_rail_detection()