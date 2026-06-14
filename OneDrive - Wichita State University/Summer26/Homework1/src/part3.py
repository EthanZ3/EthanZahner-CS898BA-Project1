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