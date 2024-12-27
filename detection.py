
import cv2
import numpy as np

def hough_circle_transform(image, min_radius, max_radius, threshold):
    # Apply gaussian blur to smooth image
    image =cv2.GaussianBlur(image, (5, 5), 0)
    # Compute sin and cos values for samples of theta
    theta_values = np.arange(0, 360, 1)
    cos_theta = np.cos(np.deg2rad(theta_values))
    sin_theta = np.sin(np.deg2rad(theta_values))

    # Perform Canny edge detection 
    edges = cv2.Canny(image, 100, 200)
    y_indexes, x_indexes = np.nonzero(edges)  # Get coordinates of all edges

    height, width = edges.shape
    
    votes = np.zeros((height, width, max_radius), dtype=np.uint64)

    # Apply Hough transform only on edge regions 
    for r in range(min_radius, max_radius):
        # create an array of possible lengths of x and y for each degree theta sampled for each radius 
        x_vals = (np.expand_dims(x_indexes, axis=1) - r * cos_theta).astype(np.int32)  
        y_vals = (np.expand_dims(y_indexes, axis=1) - r * sin_theta).astype(np.int32)  

        # ensure that x and y values from a pixel would be a valid point
        valid_indexes = (x_vals >= 0) & (x_vals < width) & (y_vals >= 0) & (y_vals < height)

        # Update votes for valid coordinates
        for rad in range(len(theta_values)): 
            # 0th Dim - edge location, 1st Dim - length of x and y per given theta 
            valid_x = x_vals[:, rad][valid_indexes[:, rad]] # Filter out invalid x vand y values
            valid_y = y_vals[:, rad][valid_indexes[:, rad]] 
            votes[valid_y, valid_x, r] += 1 # vote for a ray in the directions x, y and magnitude of r 

    detected_circles = [] # array to store circles
    region_mask = np.zeros((height, width), dtype=np.uint8)

    # normalize votes to a value between 0 and 1
    if np.amax(votes) != 0:
        votes = votes/np.amax(votes)

    # Search for circles, but only keep the largest circle in each region
    for r in range(max_radius - 1, min_radius - 1, -1):
        for y in range(height):
            for x in range(width):
                # If circle is voted enough times and the area doesnt have an existing circle, add circle
                if votes[y, x, r] >= threshold and region_mask[y, x] == 0:
                    # Mark the region as visited
                    cv2.circle(region_mask, (x, y), int(r * 1.5), 1, thickness=-1)
                    # Add circle to detected
                    detected_circles.append((x, y, r))

    return detected_circles



def extract_coin_features(image, circle):
    """
    Extract features from each detected circle.
    """
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
    coin_region = cv2.bitwise_and(image, image, mask=mask)
    hsv_coin = cv2.cvtColor(coin_region, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_coin[mask == 255]
    if hsv_values.size == 0:
        return None
    avg_hsv = np.mean(hsv_values, axis=0)
    return avg_hsv