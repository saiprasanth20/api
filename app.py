from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Load the model
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model('brain_tumor_detector.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model")

# Load model when the app starts
load_model()

def preprocess_image(image_bytes):
    """Preprocess the uploaded image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        img = preprocess_image(img_bytes)
        prediction = model.predict(img)
        probability = float(prediction[0][0])
        
        return jsonify({
            "probability": probability,
            "prediction": "Tumor Present" if probability > 0.5 else "No Tumor",
            "confidence": probability if probability > 0.5 else 1 - probability
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)