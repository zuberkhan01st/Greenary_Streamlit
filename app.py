import streamlit as st
import detectree as dtr
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from io import BytesIO

# Create directories for uploads and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_image(file_path):
    # Use detectree to process the image
    y_pred = dtr.Classifier().predict_img(file_path)

    # Calculate the percentage of detected trees
    tree_percentage = calculate_tree_percentage(y_pred)

    # Save the processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    plt.imshow(y_pred, cmap='gray')  # Use gray colormap for binary masks
    plt.axis('off')
    plt.savefig(processed_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return processed_image_path, tree_percentage

def calculate_tree_percentage(y_pred):
    # Convert y_pred to a NumPy array if it's not already
    if isinstance(y_pred, Image.Image):
        y_pred = np.array(y_pred)

    # Count white pixels (value of 255 for grayscale images)
    total_pixels = y_pred.size
    white_pixels = np.sum(y_pred == 255)  # White pixels are often represented as 255 in grayscale
    tree_percentage = (white_pixels / total_pixels) * 100

    return tree_percentage

# Streamlit app
st.title("Tree Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the image
    processed_file_path, tree_percentage = process_image(file_path)

    # Display results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Tree Percentage: {tree_percentage:.2f}%")

    # Display processed image
    with open(processed_file_path, "rb") as f:
        st.image(f.read(), caption='Processed Image', use_column_width=True)
