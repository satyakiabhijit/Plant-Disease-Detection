from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('plant_disease_model.h5')

# Define the classes for plant disease detection
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Define the root route to check if the server is up
@app.route('/')
def index():
    return "Plant Disease Detection API is running!"

# Define a route for plant disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the uploaded image file
    plant_image = request.files['file']
    
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Resize the image to the required input shape for the model
    opencv_image = cv2.resize(opencv_image, (256, 256))

    # Expand dimensions to match the input shape for the model
    opencv_image = np.expand_dims(opencv_image, axis=0)

    # Predict the disease
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]

    # Return the prediction result
    return jsonify({
        'prediction': result,
        'message': f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}."
    })

# Run the Flask app (if this is the main file being run)
if __name__ == "__main__":
    app.run(debug=True)
