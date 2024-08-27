import streamlit as st
import cv2
import numpy as np
from models.fall_detection import FallDetectionModel

# Initialize the model
fall_detector = FallDetectionModel()

# Streamlit Interface
st.title("AI-Powered Fall Detection For Home")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Detect pose and check for fall
    results = fall_detector.detect_pose(image)
    fall_status = fall_detector.check_fall(results)
    
    # Draw pose landmarks
    fall_detector.draw_pose(image, results)
    
    # Display results
    st.image(image, channels="BGR", caption="Processed Image with Pose Estimation")
    
    if "Fall detected" in fall_status:
        st.warning(fall_status)
    else:
        st.success("No Fall Detected.")
