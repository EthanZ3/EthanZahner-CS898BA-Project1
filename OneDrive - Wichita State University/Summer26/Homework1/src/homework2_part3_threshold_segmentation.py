from pathlib import Path
import cv2
import numpy as np


# input from Homework Two Part 2
NORMALIZED_IMAGE = Path("images/output/homework2/part2/normalized_color_image.png")

# Output folder for Part 3
OUTPUT_DIR = Path("images/output/homework2/part3_threshold")

# Output files
GRAYSCALE_IMAGE = OUTPUT_DIR / "normalized_grayscale.png"

OTSU_MASK = OUTPUT_DIR / "otsu_binary_mask.png"
OTSU_FOREGROUND = OUTPUT_DIR / "otsu_foreground_extraction.png"

ADAPTIVE_MASK = OUTPUT_DIR / "adaptive_binary_mask.png"
ADAPTIVE_FOREGROUND = OUTPUT_DIR / "adaptive_foreground_extraction.png"


def create_output_folder():
    # make the output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    # Load the normalized color image
    image = cv2.imread(str(image_path))
    #error handling if image cannot be found
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def make_foreground_white(mask):
    """
    Segmentation masks work best when the foreground is white and the background is black
    This checks the border and if the border is mostly white, invert he mask
    """

    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]

    border_pixels = np.concatenate([top, bottom, left, right])
    border_average = np.mean(border_pixels)

    if border_average > 127:
        mask = cv2.bitwise_not(mask)

    return mask


def extract_foreground(color_image, mask):
    # keep only the pixels where the mask is white
    foreground = cv2.bitwise_and(color_image, color_image, mask=mask)

    return foreground


def otsu_threshold_segmentation(color_image, gray_image):
    # Otsu automatically chooses one global threshold value
    _, otsu_mask = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Make sure the OBJECT is white and the BACKGROUND is black
    otsu_mask = make_foreground_white(otsu_mask)

    # Grab the foreground object from the color image
    otsu_foreground = extract_foreground(color_image, otsu_mask)

    return otsu_mask, otsu_foreground


def adaptive_threshold_segmentation(color_image, gray_image):
    #Thresholding uses local neighborhoods instead of one global value
    adaptive_mask = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2
    )

    # Make sure the OBJECT is white and the BACKGROUND is black
    adaptive_mask = make_foreground_white(adaptive_mask)

    # Grab the foreground object from the color image
    adaptive_foreground = extract_foreground(color_image, adaptive_mask)

    return adaptive_mask, adaptive_foreground


def main():
    create_output_folder()

    # Load normalized color image from Part 2
    normalized_image = load_image(NORMALIZED_IMAGE)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

    # Save the image
    cv2.imwrite(str(GRAYSCALE_IMAGE), gray_image)

    # Otsu thresholding
    otsu_mask, otsu_foreground = otsu_threshold_segmentation(
        normalized_image,
        gray_image
    )

    # Adaptive thresholding
    adaptive_mask, adaptive_foreground = adaptive_threshold_segmentation(
        normalized_image,
        gray_image
    )

    # Save Otsu results
    cv2.imwrite(str(OTSU_MASK), otsu_mask)
    cv2.imwrite(str(OTSU_FOREGROUND), otsu_foreground)

    # Save threshold results
    cv2.imwrite(str(ADAPTIVE_MASK), adaptive_mask)
    cv2.imwrite(str(ADAPTIVE_FOREGROUND), adaptive_foreground)

    print("Homework Two Part 3 complete.")
    print(f"Saved grayscale image to: {GRAYSCALE_IMAGE}")
    print(f"Saved Otsu mask to: {OTSU_MASK}")
    print(f"Saved Otsu foreground to: {OTSU_FOREGROUND}")
    print(f"Saved adaptive mask to: {ADAPTIVE_MASK}")
    print(f"Saved adaptive foreground to: {ADAPTIVE_FOREGROUND}")


if __name__ == "__main__":
    main()