from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model and its weights
model = load_model("Fer_Model.h5")

# Preprocess the image for prediction
def preprocess_image(image):
    # Resize the image to the required input shape (48x48)
    resized_image = cv2.resize(image, (48, 48))
    # Convert the image to grayscale and normalize the pixel values
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) / 255.0
    # Reshape the image to match the input shape expected by the model
    reshaped_image = np.reshape(gray_image, (1, 48, 48, 1))
    return reshaped_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['image']
        
        # Check if the file is an image
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            # Read the image using OpenCV
            img_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            # Check if the image is None (decoding failed)
            if image is None:
                return jsonify({'error': 'Failed to decode image'})

            # Preprocess the image for prediction
            processed_image = preprocess_image(image)

            # Make a prediction using the model
            prediction = model.predict(processed_image)

            # Get the index of the dominant emotion
            dominant_emotion_index = np.argmax(prediction)

            # Get the label of the dominant emotion
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            dominant_emotion = emotion_labels[dominant_emotion_index]

            # Return the predicted dominant emotion
            return jsonify({'emotion': dominant_emotion, 'message': 'Emotion detected successfully!'})

        else:
            return jsonify({'error': 'File type not allowed'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
