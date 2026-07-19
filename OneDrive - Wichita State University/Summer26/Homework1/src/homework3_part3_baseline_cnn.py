from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# start settings
SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

CLASS_NAMES = [
    "Beta",
    "Cray",
    "Discuss",
    "Gold",
    "Guppy",
    "Oscar",
]

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

SPLIT_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part2"
)

OUTPUT_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part3"
)

MODEL_FOLDER = (
    PROJECT_FOLDER
    / "models"
    / "Homework3"
)


# Training-only augmentation
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(
            0.15,
            value_range=(0.0, 1.0),
        ),
    ]
)


def load_image(filepath, label):
    """
    normalize one image.
    """

    image = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = image / 255.0

    return image, label


def create_dataset(csv_filename, training=False):
    """
    Create a dataset from a Part 2 CSV file.
    """

    csv_path = SPLIT_FOLDER / csv_filename

    dataframe = pd.read_csv(csv_path)

    # Handle either absolute or project-relative image paths
    dataframe["filepath"] = dataframe["filepath"].apply(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else PROJECT_FOLDER / value
        )
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["filepath"].values,
            dataframe["label"].astype("int32").values,
        )
    )

    if training:
        dataset = dataset.shuffle(
            len(dataframe),
            seed=SEED,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.map(
            lambda images, labels: (
                augmentation(
                    images,
                    training=True,
                ),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def build_model():
    """
     baseline CNN.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(128, 128, 3)
            ),

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),

            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(
                len(CLASS_NAMES),
                activation="softmax",
            ),
        ]
    )

    return model


def save_graphs(history):
    """
    Save accuracy and loss graphs
    """

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)

    plt.plot(
        history.history["accuracy"],
        label="Training",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation",
    )

    plt.title("Baseline CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(
        history.history["loss"],
        label="Training",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation",
    )

    plt.title("Baseline CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / "baseline_training_curves.png",
        dpi=150,
    )

    plt.close()


def main():
    """
    Train and test the baseline CNN.
    """

    tf.keras.utils.set_random_seed(SEED)

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Creating datasets...")

    train_dataset = create_dataset(
        "train_split.csv",
        training=True,
    )

    validation_dataset = create_dataset(
        "validation_split.csv"
    )

    test_dataset = create_dataset(
        "test_split.csv"
    )

    print("Building baseline CNN...")

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
    )

    print("\nTesting model...")

    test_loss, test_accuracy = model.evaluate(
        test_dataset
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save model weights
    model.save_weights(
        MODEL_FOLDER
        / "baseline_cnn.weights.h5"
    )

    # save training history
    pd.DataFrame(
        history.history
    ).to_csv(
        OUTPUT_FOLDER
        / "baseline_training_history.csv",
        index=False,
    )

    # Save test results
    results = (
        "Baseline CNN Results\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n"
    )

    (
        OUTPUT_FOLDER
        / "baseline_test_results.txt"
    ).write_text(
        results,
        encoding="utf-8",
    )

    save_graphs(history)

    print("\nTraining complete.")

    print(
        "Model weights saved in: "
        f"{MODEL_FOLDER}"
    )

    print(
        "Results saved in: "
        f"{OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()