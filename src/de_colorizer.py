import os
import cv2
import numpy as np
from PIL import Image

# --- Paths --- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "input_images")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "..", "output_images")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Adjustable Parameters --- #
RESIZE_WIDTH, RESIZE_HEIGHT = 2550, 3300  # final image size
BILATERAL_D = 5                           # filter pixel diameter
BILATERAL_SIGMA_COLOR = 75               # color smoothing strength
BILATERAL_SIGMA_SPACE = 75               # spatial smoothing strength
LAPLACIAN_KSIZE = 5                      # kernel size for edge detection
THRESH_VALUE = 10                        # binarization threshold (lower = more detail)
MORPH_KERNEL_SIZE = (1, 1)               # size of the kernel for morphology
ERODE_ITERATIONS = 1                     # how much to thin the lines

def process_image(image_path, output_path):
    """
    Processes an input image to create a coloring book-style output.

    Steps:
    - Converts the image to grayscale.
    - Applies bilateral filtering to smooth while preserving edges.
    - Uses Laplacian edge detection to extract outlines.
    - Applies thresholding to binarize the edge map.
    - Performs morphological operations to refine edges.
    - Inverts colors to get black outlines on a white background.
    - Resizes the image to printable dimensions.
    - Saves the final image to the specified output path.

    Args:
    - image_path (str): Path to the input image file.
    - output_path (str): Path where the processed image will be saved.
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")  # Error handling
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter
    smooth = cv2.bilateralFilter(
        gray, d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )

    # Laplacian edge detection
    edges = cv2.Laplacian(smooth, ddepth=cv2.CV_8U, ksize=LAPLACIAN_KSIZE)

    # Threshold to binarize edge map
    _, thresh = cv2.threshold(edges, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean and thicken edges
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(closed, kernel, iterations=ERODE_ITERATIONS)

    # Invert for coloring book style (black lines on white)
    final = cv2.bitwise_not(eroded)

    # Resize to printable size
    pil_img = Image.fromarray(final)
    pil_img = pil_img.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.LANCZOS)
    pil_img.save(output_path)

    print(f"✅ Saved: {output_path}")

def main():
    """
    Main function to process all supported image files in the input folder.

    Iterates over all .jpg, .jpeg, and .png files in the INPUT_FOLDER,
    processes each with `process_image()`, and saves results to OUTPUT_FOLDER.
    """
    
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"coloring_{filename}")
            process_image(input_path, output_path)

# --------- MAIN EXECUTION BLOCK --------------- #
if __name__ == "__main__":
    main()
