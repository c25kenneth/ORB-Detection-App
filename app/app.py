import streamlit as st
import numpy as np
from PIL import Image
import cv2
import ORB

st.title("ORB Feature Detection")
st.write("Upload two images, and get similar key points in the images.")


file_1 = st.file_uploader(label="Upload your first image")
file_2 = st.file_uploader(label="upload your second image")

button = st.button(label="Get detections!")

if file_1 is not None and file_2 is not None and button:
    image = Image.open(file_1)
    img1_array = np.array(image)
    acquiredIMG1 = cv2.imwrite(
        "img1.png", img1_array)

    image2 = Image.open(file_2)
    img2_array = np.array(image2)
    acquiredIMG2 = cv2.imwrite(
        "img2.png", img2_array)

    finalArr = ORB.detectFeatures()
    st.write("")
    st.image(finalArr)
