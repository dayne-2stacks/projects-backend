import cv2
import numpy as np
import random

def main():
    # Read the input filename from stdin
    input_filename = input().strip()

    # Load the reference and input images
    ref_image = cv2.imread('reference.png', cv2.IMREAD_GRAYSCALE)
    input_image = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded properly
    if ref_image is None:
        return
    if input_image is None:
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=7000)

    # Get keypoints and descriptors for the reference image
    kp_ref, des_ref = sift.detectAndCompute(ref_image, None)
    if des_ref is None or len(kp_ref) == 0:
        return

    # Get keypoints and descriptors for the input image
    kp_input, des_input = sift.detectAndCompute(input_image, None)
    if des_input is None or len(kp_input) == 0:
        return

    # Descriptor matching
    matches = match_descriptors(des_ref, des_input, threshold=0.7)

    # Check if enough matches are found
    if len(matches) < 4:
        print("0 0 0 0")
        return

    # Extract matched keypoints
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_input[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute transformation matrix 
    H, inliers = compute_best_transformation(src_pts, dst_pts, matches)
    if H is None:
        print("0 0 0 0")
        return

    # Filter matches and points using inliers
    matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    src_pts = src_pts[inliers]
    dst_pts = dst_pts[inliers]


    # Define bounding box in the reference image
    h_ref, w_ref = ref_image.shape[:2]
    pts_ref = np.float32([[0, 0], [w_ref - 1, 0], [w_ref - 1, h_ref - 1], [0, h_ref - 1]]).reshape(-1, 1, 2)

    # Transform bounding box to input image using transformation matrix
    pts_input_transformed = cv2.transform(pts_ref, H)

    # Compute oriented bounding box parameters
    X, Y, H_out, A = compute_obb(pts_input_transformed)

    # Ensure outputs are within the specified ranges
    X = int(round(X))
    Y = int(round(Y))
    H_out = int(round(H_out))
    A = int(round(A % 360))

    # Print the results
    print(f'{X} {Y} {H_out} {A}')

def match_descriptors(des_ref, des_input, threshold=0.25):
    """
    Match descriptors between des_ref and des_input using ratio test.
    """
    # Ensure descriptors are floats for consistency
    des_ref = des_ref.astype(np.float32)
    des_input = des_input.astype(np.float32)

    # Compute squared Euclidean distances between descriptors
    dist_matrix = np.sum(des_ref**2, axis=1, keepdims=True) + np.sum(des_input**2, axis=1) - 2 * np.dot(des_ref, des_input.T)

    # Set small negative distances to zero
    dist_matrix = np.maximum(dist_matrix, 0)

    # For each descriptor in des_ref, find the two nearest neighbors in des_input
    nn_indices_input = np.argsort(dist_matrix, axis=1)[:, :2]
    nn_dists_input = np.take_along_axis(dist_matrix, nn_indices_input, axis=1)

    # For each descriptor in des_input, find the nearest neighbor in des_ref
    nn_indices_ref = np.argmin(dist_matrix, axis=0)

    # Apply ratio test and mutual nearest neighbor check
    good_matches = []
    for i in range(len(des_ref)):
        # Indices and distances of the two nearest neighbors in des_input
        idx1 = nn_indices_input[i, 0]
        dist1 = np.sqrt(nn_dists_input[i, 0])
        dist2 = np.sqrt(nn_dists_input[i, 1])

        # Apply ratio test
        if dist1 < threshold * dist2:
            # Check if idx1 is also i's nearest neighbor
            if nn_indices_ref[idx1] == i:
                # Mutual best match
                match = cv2.DMatch(_queryIdx=i, _trainIdx=idx1, _imgIdx=0, _distance=dist1)
                good_matches.append(match)

    # Sort the matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    return good_matches


def compute_best_transformation(src_pts, dst_pts, matches, threshold=5.0, max_iterations=2000):
    """
    Compute the best affine transformation
    """
    num_points = len(matches)
    if num_points < 3:
        return None, None

    # Reshape points
    src_pts = src_pts.reshape(-1, 2)
    dst_pts = dst_pts.reshape(-1, 2)

    # Init variables
    best_inliers = None
    best_H = None
    max_inliers = 0

    #Do max_iterations number of iterations
    for _ in range(max_iterations):
        # If there are less than 3 points do nothing
        if num_points < 3:
            continue

        # ranomly pick 3 samples as indices
        idx_sample = random.sample(range(num_points), 3)

        # pick these points in the two sets
        src_sample = src_pts[idx_sample]
        dst_sample = dst_pts[idx_sample]

        # Get the affine transformation
        H = cv2.getAffineTransform(src_sample.astype(np.float32), dst_sample.astype(np.float32))

        # Resnape to perform transformation
        src_pts_expanded = src_pts.reshape(-1, 1, 2).astype(np.float32)
        # Perform Transformation
        transformed_src_pts = cv2.transform(src_pts_expanded, H)
        # Reshape to easily compute euclidean distance
        transformed_src_pts = transformed_src_pts.reshape(-1, 2)

        # Compute errors using manual distance calculation
        errors = np.sqrt(((transformed_src_pts - dst_pts) ** 2).sum(axis=1))

        # Compute inliers by removing distances too far away from expected
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        # Update best transformation if number of inliers is greater
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_H = H

    if max_inliers < 4:
        return None, None

    # Recompute transformation using all inliers
    src_inliers = src_pts[best_inliers]
    dst_inliers = dst_pts[best_inliers]

    # Compute affine transformation
    best_H, _ = cv2.estimateAffine2D(src_inliers, dst_inliers, method=cv2.LMEDS)

    return best_H, best_inliers

def compute_obb(pts_input):
    """
    Compute the Oriented Bounding Box (OBB) parameters from transformed points.
    """
    # Reshape points for processing
    pts_input = pts_input.reshape(-1, 2)

    # Compute the minimum area rectangle
    rect = cv2.minAreaRect(pts_input)

    # Extract center, size, and angle from the rectangle
    ((center_x, center_y), (width, height), angle) = rect

    # Adjust angle and height according to OpenCV conventions
    if width < height:
        H = height
        angle = angle
    else:
        H = width
        angle += 90

    # Adjust angle to be between 0 and 360 degrees
    A = (angle + 360) % 360

    # Output X, Y, H, A
    return center_x, center_y, H, A

def draw(cx, cy, h, a, img):
    sina = np.sin(np.pi*a/180.0)
    cosa = np.cos(np.pi*a/180.0)
    vx, vy = -sina*h/2.0, cosa*h/2.0
    hx, hy = cosa*0.6*h/2.0, sina*0.6*h/2.0
    cv2.line(img, (cx,cy), (int(cx+vx), int(cy+vy)), (0,255,0), 3)
    cv2.line(img, (cx,cy), (int(cx+hx), int(cy+hy)), (0,0,255), 3)
    cv2.line(img, (int(cx-vx-hx),int(cy-vy-hy)), (int(cx-vx+hx), int(cy-vy+hy)),
    (255,0,0), 3)
    cv2.line(img, (int(cx-vx+hx),int(cy-vy+hy)), (int(cx+vx+hx), int(cy+vy+hy)),
    (255,0,0), 3)
    cv2.line(img, (int(cx+vx+hx),int(cy+vy+hy)), (int(cx+vx-hx), int(cy+vy-hy)),
    (255,0,0), 3)
    cv2.line(img, (int(cx+vx-hx),int(cy+vy-hy)), (int(cx-vx-hx), int(cy-vy-hy)),
    (255,0,0), 3)


if __name__ == '__main__':
    main()