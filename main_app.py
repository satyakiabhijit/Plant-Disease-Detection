# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
try:
    model = load_model('plant_disease_model.h5')
except OSError:
    st.error("Model file not found. Please ensure 'plant_disease_model.h5' is in the correct path.")

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        try:
            # Convert the uploaded image into OpenCV format
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            st.image(opencv_image, channels="BGR")
            st.write("Original Image Shape:", opencv_image.shape)

            # Resizing the image to match model input
            opencv_image = cv2.resize(opencv_image, (256, 256))

            # Normalize the image for prediction
            opencv_image = opencv_image / 255.0  # Normalize

            # Reshaping image to include batch dimension (1, 256, 256, 3)
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            st.title(f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.warning("Please upload an image first.")
