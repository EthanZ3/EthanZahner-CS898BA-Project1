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

def describe_pipeline(filename):
    #creates text explaining what happened to the image
    name = filename.lower()
    steps = ["Original"]

    # Detect the image type from its filename
    if "grayscale" in name:
        steps.append("Grayscale")
    elif "binary" in name:
        steps.append("Binary")
    elif "hsv_value_equalized" in name:
        steps.append("HSV V Equalized")
        steps.append("BGR")
    elif "hsv" in name:
        steps.append("HSV")
    elif "cielab" in name:
        steps.append("CIELAB")
    elif "hls" in name:
        steps.append("HLS")
    else:
        steps.append("BGR")

    #Finds rotatoin and scale in the filename
    rotate_match = re.search(r"rotate_(-?\d+\.?\d*)_scale_(\d+\.?\d*)", name)

    #Finds translation in the filename
    translate_match = re.search(r"translate_x(-?\d+)_y(-?\d+)", name)

    #Finds Gaussian blur sigma in the filename
    sigma_match = re.search(r"gaussian_sigma_(\d+)_(\d+)", name)

    if rotate_match:
        angle = rotate_match.group(1)
        scale = rotate_match.group(2)
        steps.append(f"Affine(Rot:{angle}, Scale:{scale})")

    if translate_match:
        tx = translate_match.group(1)
        ty = translate_match.group(2)
        steps.append(f"Affine(Trans:[{tx},{ty}])")

    if sigma_match:
        sigma = f"{sigma_match.group(1)}.{sigma_match.group(2)}"
        steps.append(f"Gaussian Blur(σ:{sigma})")

    return " → ".join(steps)


def display_image_for_plot(image):
    # Grayscale images can be displayed as-is
    if len(image.shape) == 2:
        return image

    #Convert BGR to RGB so matplotlib shows colors correctly
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_five_image_plot(index, input_path, input_image, sobel, laplacian, canny, prewitt):
    # Get the image processing steps for the title
    pipeline = describe_pipeline(input_path.name)

    #Make alarge dark plot
    fig = plt.figure(figsize=(12, 12), facecolor="#222222")

    # add the main title
    fig.suptitle(
        f"Sample {index} Pipeline Trajectory:\n{pipeline}",
        color="cyan",
        fontsize=12,
        y=0.97
    )

    # Place images similar to the example layout
    ax_sobel = fig.add_axes([0.36, 0.68, 0.28, 0.18])
    ax_laplacian = fig.add_axes([0.05, 0.40, 0.28, 0.18])
    ax_input = fig.add_axes([0.36, 0.40, 0.28, 0.18])
    ax_canny = fig.add_axes([0.67, 0.40, 0.28, 0.18])
    ax_prewitt = fig.add_axes([0.36, 0.12, 0.28, 0.18])

    # Put all plot items in one list
    plot_items = [
        (ax_sobel, sobel, "Sobel Edge", "gray"),
        (ax_laplacian, laplacian, "Laplacian Edge", "gray"),
        (ax_input, display_image_for_plot(input_image), "Input Image", None),
        (ax_canny, canny, "Canny Edge", "gray"),
        (ax_prewitt, prewitt, "Prewitt Edge", "gray"),
    ]

    # add each image to the plot
    for ax, image, title, cmap in plot_items:
        ax.set_facecolor("#222222")
        ax.set_title(title, color="lightgray", fontsize=10)

        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)

        ax.axis("off")

    # Save the finished plot
    plot_path = PLOT_DIR / f"plot_{index:02d}_{input_path.stem}.png"
    fig.savefig(plot_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)

    return plot_path

def process_selected_subset(selected_subset):
    #stores paths to the 42 plots
    plot_paths = []

    # Process each image in the subset
    for index, input_path in enumerate(selected_subset, start=1):
        image = cv2.imread(str(input_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        # convert to grayscale for edge detection
        gray = convert_to_gray(image)

        #save the image before edge detection
        input_output_name = f"input_{index:02d}_{input_path.name}"
        input_output_path = SELECTED_INPUT_DIR / input_output_name
        cv2.imwrite(str(input_output_path), image)

        # Create edge images
        sobel = sobel_edge(gray)
        laplacian = laplacian_edge(gray)
        canny = canny_edge(gray)
        prewitt = prewitt_edge(gray)

        # Save
        cv2.imwrite(str(EDGE_DIR / "sobel" / f"sobel_{index:02d}_{input_path.name}"), sobel)
        cv2.imwrite(str(EDGE_DIR / "laplacian" / f"laplacian_{index:02d}_{input_path.name}"), laplacian)
        cv2.imwrite(str(EDGE_DIR / "canny" / f"canny_{index:02d}_{input_path.name}"), canny)
        cv2.imwrite(str(EDGE_DIR / "prewitt" / f"prewitt_{index:02d}_{input_path.name}"), prewitt)

        # Create the 5-image comparison plot
        plot_path = make_five_image_plot(
            index,
            input_path,
            image,
            sobel,
            laplacian,
            canny,
            prewitt
        )

        plot_paths.append(plot_path)

    return plot_paths


def main():
    # clear old output and make new folders
    reset_output_folders()

    # Load the 168 images from Part 2
    part2_images = load_part2_images()

    # Create 4 random subsets of 42 images
    subsets = create_four_subsets(part2_images)

    # Choose one subset
    selected_subset = subsets[SELECTED_SUBSET_NUMBER - 1]

    print(f"Loaded {len(part2_images)} images from Part 2.")
    print("Created 4 subsets of 42 images.")
    print(f"Using subset {SELECTED_SUBSET_NUMBER} for edge detection.")

    # Run edge detection and make plots
    plot_paths = process_selected_subset(selected_subset)

    

    print("\nDone.")


if __name__ == "__main__":
    main()