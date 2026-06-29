from pathlib import Path
import cv2
import numpy as np


# normalized image created in Part 2
INPUT_IMAGE = Path("images/output/homework2/part2/normalized_color_image.png")

# Part 4 output will be saved here
OUTPUT_DIR = Path("images/output/homework2/part4_kmeans")

# K values
K_VALUES = [3, 4, 5]

# After checking the output images, REMMBER TO CHANGE THESE VALUES
BEST_K = 5
BEST_CLUSTER = 1


def create_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image():
    image = cv2.imread(str(INPUT_IMAGE))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE}")

    return image


def run_kmeans(image, k):
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image so each pixel is one row of data
    pixel_values = hsv_image.reshape((-1, 3))

    #K-Means needs float32 data
    pixel_values = np.float32(pixel_values)

    # criteria tells K-Means when to stop
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    #makes the results repeatable
    cv2.setRNGSeed(898)

    # Run K-Means
    _, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Put the labels back into the shape of the image
    labels = labels.reshape(image.shape[:2])

    return labels


def save_cluster_images(image, labels, k):
    # Make a folder for this K value
    k_folder = OUTPUT_DIR / f"k_{k}"
    k_folder.mkdir(parents=True, exist_ok=True)

    # save each cluster as a mask and foreground extraction
    for cluster_id in range(k):
        # create mask
        mask = np.where(labels == cluster_id, 255, 0).astype(np.uint8)

        # use the mask to isolate part of the image
        foreground = cv2.bitwise_and(image, image, mask=mask)

        # Save the mask and foreground image
        mask_path = k_folder / f"k_{k}_cluster_{cluster_id}_mask.png"
        foreground_path = k_folder / f"k_{k}_cluster_{cluster_id}_foreground.png"

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(foreground_path), foreground)


def save_final_choice(image, labels):
    # Make the final mask using the chosen K and cluster
    mask = np.where(labels == BEST_CLUSTER, 255, 0).astype(np.uint8)

    # Extract the selected foreground
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # Save
    cv2.imwrite(str(OUTPUT_DIR / "final_kmeans_mask.png"), mask)
    cv2.imwrite(str(OUTPUT_DIR / "final_kmeans_foreground.png"), foreground)


def main():
    create_output_folder()

    image = load_image()

    all_labels = {}

    # Test K = 3, 4, and 5
    for k in K_VALUES:
        labels = run_kmeans(image, k)
        all_labels[k] = labels

        save_cluster_images(image, labels, k)

        print(f"Saved K-Means results for K = {k}")

    # save the final selected K-Means result
    save_final_choice(image, all_labels[BEST_K])

    print("Part 4 complete.")
    print(f"Best K selected: {BEST_K}")
    print(f"Best cluster selected: {BEST_CLUSTER}")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()