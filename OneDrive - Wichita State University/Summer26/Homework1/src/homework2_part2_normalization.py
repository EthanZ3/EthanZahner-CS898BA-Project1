from pathlib import Path
import cv2


# Original Homework One image
INPUT_IMAGE = Path("images/input/alien.png")

# Homework Two output folder
OUTPUT_DIR = Path("images/output/homework2/part2")

# Output image path
NORMALIZED_IMAGE = OUTPUT_DIR / "normalized_color_image.png"


def create_output_folder():
    # make the output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    #Load image using OpenCV
    image = cv2.imread(str(image_path))

    # stop if image cannot be found
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def normalize_all_color_channels(image):
    """
    OpenCV loads images as BGR, not RGB.
    this function splits into the three color channels, then applies histogram equalization to each channel separately
    merges the channels when complete
    """

    # Split the image into its three color channels
    blue, green, red = cv2.split(image)

    # Equalize each channel by itself
    equalized_blue = cv2.equalizeHist(blue)
    equalized_green = cv2.equalizeHist(green)
    equalized_red = cv2.equalizeHist(red)

    # Merges the channels back into one color image
    normalized_image = cv2.merge([
        equalized_blue,
        equalized_green,
        equalized_red
    ])

    return normalized_image


def save_image(image_path, image):
    # Save the new image
    cv2.imwrite(str(image_path), image)


def main():
    create_output_folder()

    # Load the OG image from Homework One
    original_image = load_image(INPUT_IMAGE)

    #normalize all three color channels
    normalized_image = normalize_all_color_channels(original_image)

    # Save new image
    save_image(NORMALIZED_IMAGE, normalized_image)

    print("Homework Two Part 2 complete.")
    print(f"Saved normalized image to: {NORMALIZED_IMAGE}")


if __name__ == "__main__":
    main()