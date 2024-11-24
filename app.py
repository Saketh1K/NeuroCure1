from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
from PIL import Image
import base64
import time
import os
import threading
import logging
from flask_cors import CORS
from src.preprocess import preprocess_classification
from src.inference import inference_segmentation_with_overlay, meta_pred
from src.utils import load_local_model
from src.config import *
from models.load_and_save_models import load_and_save_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to models (use environment variables or defaults)
classification_model_1_path = os.getenv('RESNET50_MODEL_PATH', 'default_resnet50_model_path.keras')
classification_model_2_path = os.getenv('CUSTOM_MODEL_PATH', 'default_custom_model_path.keras')
meta_model_path = os.getenv('META_MODEL_PATH', 'default_meta_model_path.keras')
segmentation_model_path = os.getenv('SEGMENTATION_MODEL_PATH', 'default_segmentation_model_path.keras')

# Global models (to be loaded once)
classification_model_1 = None
classification_model_2 = None
meta_model = None
segmentation_model = None

# Thread-safe flag for first run
appHasRunBefore = True
lock = threading.Lock()

# Function to load models from Google Cloud Storage (or other sources)
def from_google():
    model_paths = [
        classification_model_1_path,
        classification_model_2_path,
        segmentation_model_path,
    ]
    local_model_directory = 'models/models/'  # Local directory for saving models

    os.makedirs(local_model_directory, exist_ok=True)

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        local_model_path = os.path.join(local_model_directory, model_name)

        if os.path.exists(local_model_path):
            logger.info(f'Model {model_name} already exists. Skipping loading and saving.')
            continue

        try:
            load_and_save_model(model_path)
            logger.info(f'Model {model_name} loaded and saved successfully.')
        except Exception as e:
            logger.error(f"Failed to load and save model {model_name}: {e}")
            raise

# Function to load models into memory
def load_models():
    global classification_model_1, classification_model_2, meta_model, segmentation_model
    try:
        classification_model_1 = load_local_model(classification_model_1_path)
        classification_model_2 = load_local_model(classification_model_2_path)
        meta_model = load_local_model(meta_model_path)
        segmentation_model = load_local_model(segmentation_model_path)
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Before request: run model loading only once
@app.before_request
def firstRun():
    global appHasRunBefore
    with lock:
        if appHasRunBefore:
            logger.info("First request detected. Loading models.")
            from_google()
            load_models()
            appHasRunBefore = False

# Keep-alive thread (optional)
def keep_alive():
    while True:
        logger.info("Keep-alive task running...")
        time.sleep(60)

# Start keep-alive thread
def start_keep_alive():
    thread = threading.Thread(target=keep_alive, daemon=True)
    thread.start()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Read image from request
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Preprocess for classification
        preprocessed_image = preprocess_classification(image_np)

        # Model predictions
        resnet_preds = classification_model_1.predict(preprocessed_image)
        custom_preds = classification_model_2.predict(preprocessed_image)

        # Combine predictions for meta-model
        combined_preds = np.column_stack((resnet_preds, custom_preds))
        final_class = meta_pred(combined_preds, meta_model)
        response = {"final_class": int(final_class[0])}

        # If final_class indicates tumor, perform segmentation
        if final_class[0] != 0:
            overlayed_image = inference_segmentation_with_overlay(image_np, segmentation_model)

            # Ensure overlayed_image is valid
            if overlayed_image is None or not isinstance(overlayed_image, Image.Image):
                return jsonify({"error": "Segmentation failed"}), 500

            # Convert overlayed image to Base64
            img_io = BytesIO()
            overlayed_image.save(img_io, format='JPEG')
            img_io.seek(0)
            img_io_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            response["segment_image"] = img_io_base64

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    start_keep_alive()  # Optional keep-alive thread
    app.run(host="0.0.0.0", port=5000, debug=True)
