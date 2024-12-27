from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import random
from objectLocal import match_descriptors, compute_best_transformation, compute_obb, draw
import tempfile
from coinDetect import get_annotated_file

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)


# Specify allowed origins
ALLOWED_ORIGINS = [
    "https://portfolio-1-eight-rosy.vercel.app",
    "http://localhost:3000",  # Localhost for development
]

# Configure CORS with allowed origins
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)



def process_image(input_image_path, reference_path):
    # Load the reference and input images
    ref_image = cv2.imread(reference_path , cv2.IMREAD_GRAYSCALE)
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

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

        # Ensure H_out is a scalar
    if isinstance(H_out, (np.ndarray, np.generic)):
        H_out = float(H_out)  # Convert to scalar

    # Ensure outputs are JSON serializable
    X = int(round(X)) if isinstance(X, (float, np.float32, np.float64)) else X
    Y = int(round(Y)) if isinstance(Y, (float, np.float32, np.float64)) else Y
    H_out = int(round(H_out)) if isinstance(H_out, (float, np.float32, np.float64)) else H_out
    A = int(round(A % 360)) if isinstance(A, (float, np.float32, np.float64)) else A

    # Print the results
    print(f'{X} {Y} {H_out} {A}')

    # Annotate the image using draw
    draw(X, Y, H_out, A, input_image)

    # Save the annotated image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(temp_file.name, input_image)
    
    return temp_file.name
    

@app.route('/detect', methods=['POST'])
def detect():
    if 'input' not in request.files:
        return Response(json.dumps({"error": "Input file must be uploaded"}), status=400, mimetype="application/json")
    input_file = request.files['input']

    if input_file.filename == '':
        return Response(json.dumps({"error": "Input file must have a valid name"}), status=400, mimetype="application/json")
    
    input_path = os.path.join("/tmp", f"input_{input_file.filename}")
    input_file.save(input_path)

    try:
        # Process image
        result_path = get_annotated_file(input_path)
        if "error" in result_path:
            return jsonify(result_path), 400

        # Debug: Check if the file exists and size
        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            print(f"File not found or empty: {result_path}")
            return Response(json.dumps({"error": "Processed file not found"}), status=500, mimetype="application/json")

        print(f"Sending file: {result_path}")
        
        return send_file(result_path, mimetype='image/png', as_attachment=True)
    except Exception as e:
        error_message = {"error": str(e)}
        print("Error during processing:", error_message)
        return Response(json.dumps(error_message), status=500, mimetype="application/json")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route('/', methods=['GET'])
def index():
    return Response(json.dumps({"message": "You have successfully connected"}), status=200, mimetype="application/json")



@app.route('/process', methods=['POST'])
def process():
    try:
        if 'reference' not in request.files or 'input' not in request.files:
            return jsonify({"error": "Both reference and input files must be uploaded"}), 400

        reference_file = request.files['reference']
        input_file = request.files['input']

        if reference_file.filename == '' or input_file.filename == '':
            return jsonify({"error": "Both files must have valid names"}), 400

        input_path = os.path.join(tempfile.gettempdir(), f"input_{input_file.filename}")
        ref_path = os.path.join(tempfile.gettempdir(), f"reference_{reference_file.filename}")

        reference_file.save(ref_path)
        input_file.save(input_path)

        # Process the images
        result_path = process_image(input_path, ref_path)

        if not os.path.exists(result_path):
            return jsonify({"error": "Processed file not found"}), 500

        # Send the annotated image
        return send_file(result_path, mimetype='image/png', as_attachment=True)

    except Exception as e:
        logging.error(f"Error in /process endpoint: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary files
        for path in [input_path, ref_path]:
            if os.path.exists(path):
                os.remove(path)



@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
