from pathlib import Path
import csv
import cv2
import numpy as np


INPUT_IMAGE = Path("images/input/alien.png")
OUTPUT_DIR = Path("images/output/part2")
STATS_FILE = OUTPUT_DIR / "original_image_channel_statistics.csv"

# loading image and error handling
def ensure_output_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    return image


def save_image(filename: str, image):
    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), image)
    return output_path


def calculate_mode(channel_data):
    # since image channels are usually uint8 values from 0-255
    # np.bincount is a simple way to find the most common pixel value

    flattened = channel_data.flatten()
    counts = np.bincount(flattened, minlength=256)
    return int(np.argmax(counts))


def calculate_skew(channel_data):
    # Skew measures whether the pixel distribution leans dark or bright
    # Positive skew usually means more darker values with a tail toward bright values and
    # negative skew is the inverse

    data = channel_data.flatten().astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return 0.0

    skew = np.mean(((data - mean) / std) ** 3)
    return float(skew)


def calculate_channel_statistics(image):
    # OpenCV loads color images as BGR, not RGB. Channel order:
    # 0 = Blue
    # 1 = Green
    # 2 = Red

    channel_names = ["Blue", "Green", "Red"]
    rows = []

    for index, channel_name in enumerate(channel_names):
        channel = image[:, :, index]

        min_value = int(np.min(channel))
        max_value = int(np.max(channel))

        stats = {
            "channel": channel_name,
            "min": min_value,
            "max": max_value,
            "average": float(np.mean(channel)),
            "median": float(np.median(channel)),
            "mode": calculate_mode(channel),
            "skew": calculate_skew(channel),
            "range": max_value - min_value,
            "standard_deviation": float(np.std(channel)),
            "variance": float(np.var(channel)),
        }

        rows.append(stats)

    return rows

# This prints the statistics to the console and saves them to a csv file
def print_and_save_statistics(stats):
    print("\nOriginal Image Channel Statistics")
    print("-" * 40)

    for row in stats:
        print(f"\nChannel: {row['channel']}")
        print(f"Min: {row['min']}")
        print(f"Max: {row['max']}")
        print(f"Average: {row['average']:.2f}")
        print(f"Median: {row['median']:.2f}")
        print(f"Mode: {row['mode']}")
        print(f"Skew: {row['skew']:.4f}")
        print(f"Range: {row['range']}")
        print(f"Standard Deviation: {row['standard_deviation']:.2f}")
        print(f"Variance: {row['variance']:.2f}")

    with open(STATS_FILE, "w", newline="") as file:
        fieldnames = [
            "channel",
            "min",
            "max",
            "average",
            "median",
            "mode",
            "skew",
            "range",
            "standard_deviation",
            "variance",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    print(f"\nSaved statistics to: {STATS_FILE}")


def create_base_images(original):
    # Creates the 7 starting images:
    # 1 Original
    # 2 Grayscale
    # 3 Binary
    # 4 HSV
    # 5 CIELAB
    # 6 HLS
    # 7 HSV-normalized converted back to BGR

    base_images = {}

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    hls = cv2.cvtColor(original, cv2.COLOR_BGR2HLS)

    h, s, v = cv2.split(hsv)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])

    #Converted back to BGR for correct saving with cv2.imwrite
    # This represents the normalized RGB-color image when viewed normally
    normalized_bgr = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    base_images["01_original"] = original
    base_images["02_grayscale"] = gray
    base_images["03_binary"] = binary
    base_images["04_hsv"] = hsv
    base_images["05_cielab"] = lab
    base_images["06_hls"] = hls
    base_images["07_hsv_value_equalized_bgr"] = normalized_bgr

    for name, image in base_images.items():
        save_image(f"{name}.png", image)

    print("\nSaved 7 base images.")

    return base_images


def apply_affine_transformations(base_images):
    
    # Creates 14 transformed images total:
    # 2 unique transformations for each of the 7 base images.

    transformed_images = {}

    #Seeded random generator so results are consistent but still random
    rng = np.random.default_rng(898)

    for index, (name, image) in enumerate(base_images.items(), start=1):
        height, width = image.shape[:2]

        # Transformation 1: rotation with a unique angle and scale
        angle = rng.uniform(-45, 45)
        scale = rng.uniform(0.75, 1.25)
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

        rotated_name = f"{name}_transform_01_rotate_{angle}_scale_{scale:.2f}"
        transformed_images[rotated_name] = rotated
        save_image(f"{rotated_name}.png", rotated)

        # Transformation 2: translation with unique x/y movement
        tx = rng.integers(-50, 51)
        ty = rng.integers(-50, 51)

        translation_matrix = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])

        translated = cv2.warpAffine(image, translation_matrix, (width, height))

        translated_name = f"{name}_transform_02_translate_x{tx}_y{ty}"
        transformed_images[translated_name] = translated
        save_image(f"{translated_name}.png", translated)

    print("Saved 14 affine-transformed images.")

    return transformed_images


def apply_gaussian_blurs(all_images):

    # applies 7 gaussian blur sigma levels to each of the 21 existing images
    # 21 images x 7 sigma levels = 147 blurred images, 21 existing + 147 blurred = 168 total images

    sigma_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    count = 0

    for name, image in all_images.items():
        for sigma in sigma_levels:
            blurred = cv2.GaussianBlur(
                image,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma
            )

            sigma_text = str(sigma).replace(".", "_")
            filename = f"{name}_gaussian_sigma_{sigma_text}.png"

            save_image(filename, blurred)
            count += 1

    print(f"Saved {count} Gaussian-blurred images.")


def main():
    ensure_output_folder()

    original = load_image(INPUT_IMAGE)

    stats = calculate_channel_statistics(original)
    print_and_save_statistics(stats)

    base_images = create_base_images(original)
    transformed_images = apply_affine_transformations(base_images)

    all_images = {}
    all_images.update(base_images)
    all_images.update(transformed_images)

    print(f"\nImages before Gaussian blur: {len(all_images)}")
    apply_gaussian_blurs(all_images)

    print("\nDone.")
    print("Expected final image count:")
    print("7 base images + 14 transformed images + 147 blurred images = 168 images")


if __name__ == "__main__":
    main()