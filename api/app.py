from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os

from model_loader import model_loader
from config import CLASSES, IMG_SIZE, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load model at startup
model = model_loader.load()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Open image
    img = Image.open(image_file).convert('RGB')
    
    # Resize to 224x224
    img = img.resize(IMG_SIZE)
    
    # Convert to array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # ResNet50 preprocessing
    img_array = preprocess_input(img_array)
    
    return img_array


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'ResNet50',
        'version': '1.0',
        'classes': CLASSES
    })


@app.route('/predict', methods=['POST'])
def predict():
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Preprocess image
        img_array = preprocess_image(file)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_class_idx])
        
        # Format probabilities
        probabilities = {
            cls: float(predictions[0][i])
            for i, cls in enumerate(CLASSES)
        }
        
        # Create response
        response = {
            'success': True,
            'prediction': CLASSES[pred_class_idx],
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': CLASSES,
        'count': len(CLASSES)
    })


if __name__ == '__main__':
    print("Brain Tumor Classification API")
    print(f"Model: ResNet50")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Starting server...")
    
    app.run(host='0.0.0.0', port=5000, debug=False)