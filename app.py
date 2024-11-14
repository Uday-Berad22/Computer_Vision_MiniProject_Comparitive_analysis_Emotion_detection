from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load both models at startup
try:
    cnn_model = load_model('C:/Project/CV_Project/CNN/Face_Emotion_Recognition_Machine_Learning/cnn.h5')
    vgg_model = load_model('C:/Project/CV_Project/CNN/Face_Emotion_Recognition_Machine_Learning/VGG/vgg2.h5')
except Exception as e:
    print(f"Error loading models: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cnn_preprocess_image(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.reshape(image, (1, target_size[0], target_size[1], 1))
    return image

def vgg_preprocess_image(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.reshape(image, (1, target_size[0], target_size[1], 3))
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # CNN Prediction
            cnn_processed = cnn_preprocess_image(filepath)
            cnn_predictions = cnn_model.predict(cnn_processed)
            cnn_class = np.argmax(cnn_predictions[0])
            cnn_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            cnn_emotion = cnn_classes[cnn_class]
            cnn_confidence = float(cnn_predictions[0][cnn_class])

            # VGG Prediction
            vgg_processed = vgg_preprocess_image(filepath)
            vgg_predictions = vgg_model.predict(vgg_processed)
            vgg_class = np.argmax(vgg_predictions[0])
            vgg_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            vgg_emotion = vgg_classes[vgg_class]
            vgg_confidence = float(vgg_predictions[0][vgg_class])
            
            response = {
                'filename': filename,
                'filepath': f'/static/uploads/{filename}',
                'cnn_prediction': {
                    'emotion': cnn_emotion,
                    'confidence': f'{cnn_confidence:.2%}'
                },
                'vgg_prediction': {
                    'emotion': vgg_emotion,
                    'confidence': f'{vgg_confidence:.2%}'
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)