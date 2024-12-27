import cv2
import numpy as np
import joblib
import sys
from detection import hough_circle_transform, extract_coin_features

# Load the trained SVM model and scaler
penny_svm_model = joblib.load('penny_svm_model.pkl')
scaler = joblib.load('penny_scaler.pkl')

def classify_coin(image, circle):
    radius = circle[2]
    # Extract features for the coin
    features = extract_coin_features(image, circle)
    if features is None:
        return "Unknown"
    # Normalize features
    features = scaler.transform([features])
    # Predict using the SVM model 
    is_penny = penny_svm_model.predict(features)[0]
    if is_penny == 1:
        return "Penny"
    else:
        # If not a penny, classify based on radius
        if radius in range(20, 23):  # Nickel radius range
            return "Nickel"
        elif radius in range(18, 20):  # Dime radius range
            return "Dime"
        elif radius in range(23, 27):  # Quarter radius range
            return "Quarter"
        else:
            return "Unknown"

def get_annotated_file(input_path):
    # Read the image file name from standard input
    image_file = input_path

    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not read image {image_file}")
        sys.exit(1)

    # Downsample by a factor of 5
    height, width = image.shape[:2]
    new_width = width // 5
    new_height = height // 5

    # Resize the image
    downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Define minimum and maximum radius for downsampled images and a voting threshold
    min_radius = 18
    max_radius = 27
    threshold = 0.64

    # Find the minimum and maximum pixel values
    min_value = np.min(downsampled_image)
    max_value = np.max(downsampled_image)

    # Apply normalization to increase contrast
    normalized_image = (((downsampled_image - min_value) /
                         (max_value - min_value)) * 255).astype(np.uint8)

    # Detect circles
    circles = hough_circle_transform(
        normalized_image, min_radius, max_radius, threshold)

    print(f"{len(circles)} circles detected.")

    # Annotate the image
    for circle in circles:
        # Adjust circle coordinates back to the original image size
        x_coord = circle[0] * 5
        y_coord = circle[1] * 5
        radius = circle[2] * 5

        # Classify the coin
        coin_type = classify_coin(downsampled_image, circle)

        # Draw the circle on the original image
        cv2.circle(image, (int(x_coord), int(y_coord)), int(radius), (0, 255, 0), 2)

        # Add label text next to the circle
        label = f"{coin_type}"
        cv2.putText(image, label, (int(x_coord) + 10, int(y_coord) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save the annotated image
    output_file = "annotated_coins.png"
    cv2.imwrite(output_file, image)
    print(f"Annotated image saved as {output_file}")

    
    return output_file
