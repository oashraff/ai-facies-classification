from flask import Flask, render_template, request, jsonify
from datetime import datetime
import threading
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from resnet34_model import UNet34
from resnet50_model import UNet50
from inception_model import UNetInception

app = Flask(__name__)

# Global variables to hold the models and a flag to indicate if they're loaded
RESNET34_MODEL = None
RESNET50_MODEL = None
INCEPTION_MODEL = None
models_loaded = False

def load_models():
    global RESNET34_MODEL, RESNET50_MODEL, INCEPTION_MODEL, models_loaded
    try:
        # Load ResNet34
        resnet34_model = UNet34()
        resnet34_model.load_state_dict(torch.load('models/resnet34.pth', map_location=torch.device('cpu')))
        resnet34_model.eval()

        # Load ResNet50
        resnet50_model = UNet50(in_channels=1, out_channels=6)
        resnet50_model.load_state_dict(torch.load('models/resnet50.pth', map_location=torch.device('cpu')))
        resnet50_model.eval()

        # Load InceptionV3
        inception_model = UNetInception(in_channels=1, out_channels=6)
        inception_model.load_state_dict(torch.load('models/inceptionv3.pth', map_location=torch.device('cpu')))
        inception_model.eval()

        RESNET34_MODEL = resnet34_model
        RESNET50_MODEL = resnet50_model
        INCEPTION_MODEL = inception_model
        models_loaded = True
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False

@app.before_first_request
def initialize_models():
    # Start a background thread to load models so that the app starts quickly
    thread = threading.Thread(target=load_models)
    thread.start()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((99, 99)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models are still loading. Please try again later.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    model_name = request.form.get('model', 'resnet50')
    
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('L')
        img_tensor = transform(img).unsqueeze(0)
        
        if model_name == 'resnet50':
            model = RESNET50_MODEL
        elif model_name == 'resnet34':
            model = RESNET34_MODEL
        elif model_name == 'inceptionv3':
            model = INCEPTION_MODEL
        else:
            return jsonify({'error': 'Invalid model selection'}), 400
        
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted = outputs.argmax(dim=1)
            probabilities = F.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].mean().item()
        
        prediction_map = predicted[0].cpu().numpy()
        classes = ['Upper North Sea', 'Middle North Sea', 'Lower North Sea', 
                   'Rijnland/Chalk', 'Scruff', 'Zechstein']
        
        unique, counts = np.unique(prediction_map, return_counts=True)
        class_percentages = dict(zip([classes[i] for i in unique],
                               (counts / counts.sum() * 100).tolist()))
        
        return jsonify({
            'success': True,
            'prediction': class_percentages,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    # Lightweight endpoint to verify the app is up
    return jsonify({"status": "ok"}), 200

@app.route('/')
def home():
    return render_template('index.html', year=datetime.now().year)

@app.route('/model')
def model():
    return render_template('model.html', year=datetime.now().year)

@app.route('/about')
def about():
    return render_template('about.html', year=datetime.now().year)

@app.route('/documentation')
def documentation():
    return render_template('documentation.html', year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)
