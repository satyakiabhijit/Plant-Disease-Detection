import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('plant_disease_model.h5')

# Define class names for predictions
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust']


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Read the image in the file
    img = Image.open(file.stream)

    # Resize and preprocess the image
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    result = CLASS_NAMES[np.argmax(prediction)]

    return jsonify({'result': result})


if __name__ == "__main__":
    app.run(debug=True)
