from pathlib import Path
import re
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


#folder with the 168 images from Part 2
PART2_DIR = Path("images/output/part2")

# Main output folder for Part 3
PART3_DIR = Path("images/output/part3")

# Part 3 output folders
SELECTED_INPUT_DIR = PART3_DIR / "selected_inputs"
EDGE_DIR = PART3_DIR / "edges"
PLOT_DIR = PART3_DIR / "plots"
README_PLOT_DIR = PLOT_DIR / "readme_samples"
SUBSET_DIR = PART3_DIR / "subsets"

# Random seed makes the random choices repeatable
RANDOM_SEED = 898

# Pick which subset to use: 1, 2, 3, or 4
SELECTED_SUBSET_NUMBER = 1


def reset_output_folders():
    # Delete old Part 3 output if it exists
    if PART3_DIR.exists():
        shutil.rmtree(PART3_DIR)

    # Recreate all needed folders
    SELECTED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    README_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    # Make folders for each edge detection method
    for technique in ["sobel", "laplacian", "canny", "prewitt"]:
        (EDGE_DIR / technique).mkdir(parents=True, exist_ok=True)


def load_part2_images():
    # Get all png images from Part 2
    image_paths = sorted(PART2_DIR.glob("*.png"))

    # Part 2 should have exactly 168 images
    if len(image_paths) != 168:
        raise ValueError(
            f"Expected 168 Part 2 images in {PART2_DIR}, but found {len(image_paths)}."
        )

    return image_paths


def create_four_subsets(image_paths):
    # Shuffle the Part 2 images randomly
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    # Split into 4 groups of 42 images
    subsets = [
        shuffled[0:42],
        shuffled[42:84],
        shuffled[84:126],
        shuffled[126:168],
    ]

    # Save the file names for each subset
    for i, subset in enumerate(subsets, start=1):
        subset_file = SUBSET_DIR / f"subset_{i}.txt"

        with open(subset_file, "w") as file:
            for path in subset:
                file.write(str(path) + "\n")

    return subsets

# edge detection methods rely on brightness, not color, so it makes sense to convert them to grayscale first
# then we have to reconvert the edge result back into normal image values for saving
def convert_to_gray(image):
    # if image is already grayscale, return it
    if len(image.shape) == 2:
        return image

    #convert BGR image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_to_uint8(image):
    # Convert edge result into normal 0-255 image values
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def sobel_edge(gray):
    # Detect horizontal edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Detect vertical edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine both edges
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return normalize_to_uint8(magnitude)


def laplacian_edge(gray):
    # detect edges based on quick brightness changes
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    return normalize_to_uint8(np.absolute(laplacian))

def canny_edge(gray):
    # Use the image median to choose Canny thresholds
    median = np.median(gray)

    lower = int(max(0, 0.67 * median))
    upper = int(min(255, 1.33 * median))

    # Backup threshholds if the automatic values are bad
    if lower == upper:
        lower = 50
        upper = 150

    return cv2.Canny(gray, lower, upper)


def prewitt_edge(gray):
    #Prewitt horizontal edge kernel
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Prewitt vertical edge kernel
    kernel_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ], dtype=np.float32)

    #  Apply Prewitt filters
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    # Combine both directions
    magnitude = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

    return normalize_to_uint8(magnitude)