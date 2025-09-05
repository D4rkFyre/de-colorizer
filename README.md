# Image De-Colorizer Studio

This project contains two Python scripts that convert color images into printable black-and-white coloring book pages. One script runs from the command line, and the other provides a real-time web interface using Streamlit.

## Folder Structure

```
DE_COLORIZER/
 ─ input_images/    # Folder for input images
 ─ output_images/   # Folder where processed images will be saved
 ─ src/                  # Contains both Python scripts
   ─ de_colorizer.py
   ─ streamlit_de_colorizer.py
 ─ requirements.txt      # List of Python packages used
 ─ README.md             # This documentation file
 ```

## Scripts

### de_colorizer.py

- Location: src/de_colorizer.py
- Purpose: Automatically processes all .jpg, .jpeg, and .png files in the input_images folder and saves black-and-white "coloring book" versions to the output_images folder.
- Libraries used: OpenCV, NumPy, Pillow
- To run:
  python src/de_colorizer.py

### streamlit_de_colorizer.py

- Location: src/streamlit_de_colorizer.py
- Purpose: A web app for uploading an image, adjusting parameters, and viewing real-time results. Users can download the processed image as PNG or PDF.
- Libraries used: Streamlit, OpenCV, Pillow, NumPy
- To run:
  streamlit run src/streamlit_de_colorizer.py

## Setup Instructions

1. Clone or download the project folder.
2. Optional: Create and activate a virtual environment.
   python -m venv wk4env
   source wk4env/bin/activate     (Linux/Mac)
   wk4env\Scripts\activate.bat    (Windows)
3. Install dependencies from requirements.txt.
   pip install -r requirements.txt

## How to Use

- For the basic script, place images in input_images and run de_colorizer.py. Output will appear in output_images.
- For the Streamlit app, run the script and use the interface to upload an image. Customize settings and download results.

## Notes

- Images are resized to 2550x3300 pixels for 8.5 x 11 inch paper.
- Bilateral filtering smooths the image while keeping edges sharp.
- Laplacian edge detection is used to create outlines.
- Thresholding and inversion make the lines black on a white background.
- Morphological operations clean and enhance line quality.
- The Streamlit app includes a preset mode and advanced options for custom tuning.
