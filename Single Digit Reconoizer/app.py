import flask
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps # Pillow for image processing
import base64
import io
import os
import re # To process the base64 string

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
MODEL_FILENAME = 'my_digit_model.keras'
model = None

def load_keras_model():
    """Load the Keras model."""
    global model
    if os.path.exists(MODEL_FILENAME):
        try:
            model = keras.models.load_model(MODEL_FILENAME)
            print(f"* Model '{MODEL_FILENAME}' loaded successfully.")
        except Exception as e:
            print(f"* Error loading model: {e}")
            # Handle error appropriately, maybe exit or disable prediction
            model = None
    else:
        print(f"* Error: Model file not found at '{MODEL_FILENAME}'")
        model = None

# --- Image Preprocessing ---
def preprocess_image_data(image_data):
    """Preprocesses image data received from the canvas."""
    try:
        # Decode base64 string
        # Remove header: "data:image/png;base64,"
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        img_bytes = base64.b64decode(img_str)

        # Open image using Pillow
        img = Image.open(io.BytesIO(img_bytes))

        # 1. Convert to grayscale AND ensure it has an alpha channel for background handling
        img = img.convert('LA') # Convert to Luminance + Alpha

        # 2. Create a white background image
        bg = Image.new("LA", img.size, (255, 255)) # White background fully opaque
        # Paste the drawing (which has transparency) onto the white background
        bg.paste(img, (0, 0), img)

        # 3. Convert back to grayscale (Luminance only) - now background is white
        img = bg.convert('L')

        # 4. Invert colors (MNIST has white digit on black bg)
        # Since canvas drawing is black on white, we invert
        img = ImageOps.invert(img)

        # 5. Resize to 28x28 pixels (MNIST size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 6. Convert image to NumPy array
        img_array = np.array(img)

        # 7. Normalize pixel values from 0-255 to 0.0-1.0
        img_array = img_array.astype('float32') / 255.0

        # 8. Reshape for the model (add batch dimension and channel dimension if needed)
        # Model expects (batch_size, height, width) or (batch_size, height, width, channels)
        # Our simple dense model expects flattened (batch_size, 784)
        # Keras handles Flatten layer, so input shape (1, 28, 28) is fine.
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 28, 28)

        return img_array

    except Exception as e:
        print(f"* Error processing image data: {e}")
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive drawing data, preprocess, predict, and return result."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        image_b64 = data['image']

        processed_image = preprocess_image_data(image_b64)

        if processed_image is not None:
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_digit = int(np.argmax(predictions[0])) # Convert numpy int64 to Python int
            confidence = float(np.max(predictions[0]) * 100) # Convert numpy float32 to Python float

            print(f"* Prediction: {predicted_digit}, Confidence: {confidence:.2f}%")
            return jsonify({
                'prediction': predicted_digit,
                'confidence': round(confidence, 2)
            })
        else:
            return jsonify({'error': 'Image processing failed'}), 400

    except Exception as e:
        print(f"* Error during prediction request: {e}")
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    print("* Loading Keras model and starting Flask server...")
    load_keras_model() # Load the model
    # Use host='0.0.0.0' to make it accessible on your network (optional)
    # Use debug=True for development (auto-reloads), False for production
    app.run(host='127.0.0.1', port=5000, debug=True)
