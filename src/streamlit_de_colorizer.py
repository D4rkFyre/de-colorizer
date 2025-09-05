import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

# --- Session State Initialization --- #
def initialize_session_state():
    # Default values
    defaults = {
        "enable_spatial_smoothness": False,
        "threshold": 50,
        "bilateral_d": 9,
        "sigma_color": 75,
        "sigma_space": 75,
        "invert": False,
        "grayscale": False,
        "show_advanced": False,
        "enable_coloring_mode": False
    }
    # If any keys are not already in session_state, add them
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Function to initialize session state
initialize_session_state()

# --- Layout --- #
st.set_page_config(page_title="De-Colorizer Studio", layout="wide")
st.title("Image De-Colorizer Studio")

# --- Upload Image --- #
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Read image into memory, store its name
if uploaded_file:
    uploaded_file.seek(0)
    st.session_state.uploaded_bytes = uploaded_file.read()
    st.session_state.base_filename = os.path.splitext(uploaded_file.name)[0]

# --- Main Interface Columns --- #
# Divide screen into 2 columns
uploaded_bytes = st.session_state.get("uploaded_bytes", None)
col1, col2 = st.columns([1, 2])

# Left column: controls and options
with col1:
    if uploaded_bytes:
        st.subheader("Original Image")
        
        # Convert upload into an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Show original image
        st.image(Image.fromarray(img_rgb), caption="Uploaded Image", width=300)

        # Option to enable all default parameters to make a basic coloring book page
        st.checkbox("Coloring Book Page", key="enable_coloring_mode")

        # If coloring book mode is not enabled, show advanced controls toggle
        if not st.session_state.enable_coloring_mode:
            st.checkbox("Advanced Options", key="show_advanced")

            # Advanced tuning toggles for when advanced options are toggled
            if st.session_state.show_advanced:
                st.checkbox("Convert to Grayscale", key="grayscale")
                st.checkbox("Invert Image Colors", key="invert")
                st.checkbox("Enable Parameter Sliders", key="show_parameters")

                # Additional parameter sliders for further fine-tuning
                if st.session_state.show_parameters:
                    st.slider("Edge Detection Sensitivity",
                            0, 100, value=st.session_state.threshold, key="threshold",
                            help="Lower values detect more edges. Higher values focus on strong edges.")

                    st.slider("Edge Softness",
                            1, 15, value=st.session_state.bilateral_d, key="bilateral_d",
                            help="Higher values blur edges more, resulting in softer outlines.")

                    st.slider("Color Detail Preservation",
                            0, 150, value=st.session_state.sigma_color, key="sigma_color",
                            help="Higher values preserve more details in textured areas.")

                    st.checkbox("Enable Spatial Smoothness", key="enable_spatial_smoothness")

        # If image has been processed, show download buttons
        if "processed_result" in st.session_state:
            png_bytes = st.session_state.png_bytes
            pdf_bytes = st.session_state.pdf_bytes
            base_filename = st.session_state.base_filename

            # Download as PNG
            st.download_button("Download PNG", data=png_bytes,
                               file_name=f"coloring_{base_filename}.png",
                               mime="image/png", use_container_width=True)

            # Download as PDF
            st.download_button("Download PDF", data=pdf_bytes,
                               file_name=f"coloring_{base_filename}.pdf",
                               mime="application/pdf", use_container_width=True)

# --- Right Column: Image preview and Processing
with col2:
    if uploaded_bytes:
        # Decode image for processing
        file_bytes = np.asarray(bytearray(uploaded_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Apply Coloring Book Mode Defaults (grayscale, invert colors, bilateral filter, edge detection)
        if st.session_state.enable_coloring_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=0)
            output = cv2.Laplacian(output, ddepth=cv2.CV_8U, ksize=5)
            _, output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
            output = cv2.bitwise_not(output)

        # --- Advanced tuning mode --- #
        else:
            # Grayscale toggle
            if st.session_state.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            output = img.copy()

            # Apply filters/edge detection based on parameters
            if st.session_state.show_advanced and st.session_state.show_parameters:
                if st.session_state.grayscale and len(output.shape) == 3:
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

                sigma_space = 2 if st.session_state.enable_spatial_smoothness else 0
                output = cv2.bilateralFilter(output, d=st.session_state.bilateral_d,
                                             sigmaColor=st.session_state.sigma_color,
                                             sigmaSpace=sigma_space)
                output = cv2.Laplacian(output, ddepth=cv2.CV_8U, ksize=5)
                _, output = cv2.threshold(output, st.session_state.threshold, 255, cv2.THRESH_BINARY)

            # Invert color toggle
            if st.session_state.invert:
                output = cv2.bitwise_not(output)

        # Convert processed image to streamlit display
        result_pil = Image.fromarray(output if len(output.shape) == 2 else output.astype(np.uint8))
        st.image(result_pil, caption="Processed Preview", use_container_width=True)

        # Resize image to 8.5 x 11 for printing or saving
        result_resized = result_pil.resize((2550, 3300), Image.LANCZOS)
        
        # Save PNG version
        buf = BytesIO()
        result_resized.save(buf, format="PNG")
        
        # Save PDF version
        pdf_buf = BytesIO()
        result_resized.convert("RGB").save(pdf_buf, format="PDF")

        # Store results/content into session state
        st.session_state.processed_result = result_resized
        st.session_state.png_bytes = buf.getvalue()
        st.session_state.pdf_bytes = pdf_buf.getvalue()

'''
    Run this script from the terminal with this command:

    streamlit run ./src/streamlit_de_colorizer.py
'''
