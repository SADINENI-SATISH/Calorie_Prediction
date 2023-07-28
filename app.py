import pandas as pd
import numpy as np
import streamlit as st
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from tensorflow import keras
from PIL import Image
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib'
import cv2
import ast

model = keras.models.load_model('best_our_model.h5')

def predict_class(img):
    prediction = model.predict(img)
    return prediction

def process_image(img):
    if img is not None:
        image = Image.open(img)
        img = np.array(image)
        img = cv2.resize(img, (224, 224))  # Resize the image to 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        img = img.reshape(1, 224, 224, 3)
        return img

def main():
    st.title("Calorie Prediction")
    img = st.file_uploader('Insert your image', type=['png'])
    img = process_image(img)
    result = ""

    # Background image
    with open('calorie.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Predict"):
        result = predict_class(img)
        classes_file = open("Calories.txt", "r")
        classes = ast.literal_eval(classes_file.read())
        classes_file.close()
        predicted_classes = np.argmax(result)
        result = classes[predicted_classes]

    st.success('The sign is $ {}'.format(result))

if __name__ == '__main__':
    main()
