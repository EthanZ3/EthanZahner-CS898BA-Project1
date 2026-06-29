from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


# input images
ORIGINAL_IMAGE = Path("images/input/alien.png")
NORMALIZED_IMAGE = Path("images/output/homework2/part2/normalized_color_image.png")

# Ground truth 
GROUND_TRUTH_MASK = Path("images/ground_truth_mask.png")

# Segmentation masks from previous parts
OTSU_MASK = Path("images/output/homework2/part3_threshold/otsu_binary_mask.png")
ADAPTIVE_MASK = Path("images/output/homework2/part3_threshold/adaptive_binary_mask.png")
KMEANS_MASK = Path("images/output/homework2/part4_kmeans/final_kmeans_mask.png")

# Output folder
OUTPUT_DIR = Path("images/output/homework2/part5_evaluation")

# Output files
METRICS_CSV = OUTPUT_DIR / "segmentation_metrics.csv"
COMPARISON_PLOT = OUTPUT_DIR / "segmentation_comparison_plot.png"


def create_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_color_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")

    #mask is black and white
    binary_mask = mask > 127

    return binary_mask


def calculate_iou(ground_truth, prediction):
    # Intersection = pixels that are white in both masks
    intersection = np.logical_and(ground_truth, prediction).sum()

    # Union = pixels that are white in either mask
    union = np.logical_or(ground_truth, prediction).sum()

    if union == 0:
        return 0

    return intersection / union


def calculate_dice(ground_truth, prediction):
    # Intersection = pixels that are white in both masks
    intersection = np.logical_and(ground_truth, prediction).sum()

    # Total white pixels in both masks
    total = ground_truth.sum() + prediction.sum()

    if total == 0:
        return 0

    return (2 * intersection) / total


def save_metrics(results):
    with open(METRICS_CSV, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "iou", "dice"]
        )

        writer.writeheader()
        writer.writerows(results)


def mask_to_image(mask):
    # convert True/False mask back to black/white image
    return (mask.astype(np.uint8)) * 255


def make_comparison_plot(original, normalized, ground_truth, otsu, adaptive, kmeans):
    # Convert color images from BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)

    images = [
        (original_rgb, "Original Image", None),
        (normalized_rgb, "Normalized Image", None),
        (mask_to_image(ground_truth), "Ground Truth Mask", "gray"),
        (mask_to_image(otsu), "Otsu Mask", "gray"),
        (mask_to_image(adaptive), "Adaptive Mask", "gray"),
        (mask_to_image(kmeans), "K-Means Mask", "gray"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))

    for ax, (image, title, cmap) in zip(axes, images):
        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=150)
    plt.close()


def main():
    create_output_folder()

    # load images
    original = load_color_image(ORIGINAL_IMAGE)
    normalized = load_color_image(NORMALIZED_IMAGE)

    # Load masks
    ground_truth = load_mask(GROUND_TRUTH_MASK)
    otsu = load_mask(OTSU_MASK)
    adaptive = load_mask(ADAPTIVE_MASK)
    kmeans = load_mask(KMEANS_MASK)

    # Calculate scores
    results = [
        {
            "method": "Otsu",
            "iou": calculate_iou(ground_truth, otsu),
            "dice": calculate_dice(ground_truth, otsu)
        },
        {
            "method": "Adaptive",
            "iou": calculate_iou(ground_truth, adaptive),
            "dice": calculate_dice(ground_truth, adaptive)
        },
        {
            "method": "K-Means",
            "iou": calculate_iou(ground_truth, kmeans),
            "dice": calculate_dice(ground_truth, kmeans)
        }
    ]

    # Print scores
    print("Part 5 Segmentation Scores")
    print("--------------------------")

    for result in results:
        print(f"{result['method']}:")
        print(f"IoU:  {result['iou']:.4f}")
        print(f"Dice: {result['dice']:.4f}")
        print()

    # Save scores to CSV
    save_metrics(results)

    # Create comparison plot
    make_comparison_plot(
        original,
        normalized,
        ground_truth,
        otsu,
        adaptive,
        kmeans
    )

    print(f"Saved metrics to: {METRICS_CSV}")
    print(f"Saved comparison plot to: {COMPARISON_PLOT}")


if __name__ == "__main__":
    main()