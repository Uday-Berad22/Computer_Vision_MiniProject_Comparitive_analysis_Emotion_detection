# from flask import Flask, render_template, request, jsonify
# import os
# from werkzeug.utils import secure_filename
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from flask_cors import CORS

# app = Flask(__name__, 
#     template_folder='templates',
#     static_folder='static'
# )
# CORS(app)

# # Configure upload folder and allowed extensions
# UPLOAD_FOLDER = 'static/uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Placeholder for your model loading functions
# def load_cnn_model():
#     # Replace with your actual CNN model loading code
#     model = load_model('C:/Project/CV_Project/CNN/Face_Emotion_Recognition_Machine_Learning/cnn.h5')
#     return model

# def load_vgg_model():
#     # Replace with your actual VGG model loading code
    
#     model = load_model('C:/Project/CV_Project/CNN/Face_Emotion_Recognition_Machine_Learning/VGG/vgg2.h5')
#     return model

# # Load models at startup
# cnn_model = load_cnn_model()
# vgg_model = load_vgg_model()

# def preprocess_image(image_path, target_size=(224, 224)):
#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     files = request.files.getlist('file')
#     results = []
    
#     for file in files:
#         if file.filename == '':
#             continue
            
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#             # Preprocess image
#             processed_image = preprocess_image(filepath)
            
#             # Get predictions
#             cnn_pred = cnn_model.predict(processed_image)
#             vgg_pred = vgg_model.predict(processed_image)
            
#             # Get emotion labels and probabilities
#             emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']  # Add your actual emotion labels
#             cnn_emotion = emotions[np.argmax(cnn_pred)]
#             vgg_emotion = emotions[np.argmax(vgg_pred)]
            
#             cnn_accuracy = float(np.max(cnn_pred))
#             vgg_accuracy = float(np.max(vgg_pred))
            
#             results.append({
#                 'filename': filename,
#                 'filepath': f'/static/uploads/{filename}',
#                 'cnn_emotion': cnn_emotion,
#                 'cnn_accuracy': f'{cnn_accuracy:.2%}',
#                 'vgg_emotion': vgg_emotion,
#                 'vgg_accuracy': f'{vgg_accuracy:.2%}'
#             })
    
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
try:
    model = load_model('C:/Project/CV_Project/CNN/Face_Emotion_Recognition_Machine_Learning/cnn.h5')
except:
    print("Error loading model")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.reshape(image, (1, target_size[0], target_size[1], 1))
    return image

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'})
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"Processing file: {filepath}")
            
            # Debug prints
            processed_image = preprocess_image(filepath)
            print(f"Image shape after preprocessing: {processed_image.shape}")
            
            predictions = model.predict(processed_image)
            print(f"Raw predictions: {predictions}")
            
            predicted_class = np.argmax(predictions[0])
            print(f"Predicted class index: {predicted_class}")
            
            classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            emotion = classes[predicted_class]
            confidence = float(predictions[0][predicted_class])
            
            print(f"Emotion: {emotion}, Confidence: {confidence}")
            
            response = {
                'filename': filename,
                'filepath': f'/static/uploads/{filename}',
                'emotion': emotion,
                'confidence': f'{confidence:.2%}'
            }
            print(f"Sending response: {response}")
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
    
    
    <!DOCTYPE html>
<html>
  <head>
    <title>Emotion Detection</title>
    <script src="https://unpkg.com/dropzone@5/dist/min/dropzone.min.js"></script>
    <link
      rel="stylesheet"
      href="https://unpkg.com/dropzone@5/dist/min/dropzone.min.css"
      type="text/css"
    />
  </head>
  <body>
    <form action="/predict" class="dropzone" id="my-dropzone">
      <div class="fallback">
        <input name="file" type="file" />
      </div>
    </form>
    <div id="results"></div>

    <script>
      Dropzone.options.myDropzone = {
        paramName: "file",
        maxFilesize: 2,
        acceptedFiles: ".jpg,.jpeg,.png",
        init: function () {
          this.on("success", function (file, response) {
            console.log("Response received:", response);

            if (response.error) {
              console.error("Error:", response.error);
              return;
            }

            const resultsDiv = document.getElementById("results");
            resultsDiv.innerHTML += `
                        <div>
                            <img src="${response.filepath}" width="200">
                            <p>Emotion: ${response.emotion || "Unknown"}</p>
                            <p>Confidence: ${response.confidence || "N/A"}</p>
                        </div>
                    `;
          });

          this.on("error", function (file, errorMessage) {
            console.error("Upload error:", errorMessage);
          });
        },
      };
    </script>
  </body>
</html>
