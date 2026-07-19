from pathlib import Path
import json

import pandas as pd
import tensorflow as tf


# Basic settings
SEED = 42
IMAGE_SIZE = (128, 128)
EPOCHS = 8

CLASS_NAMES = [
    "Beta",
    "Cray",
    "Discuss",
    "Gold",
    "Guppy",
    "Oscar",
]

# Values tested during grid search
LEARNING_RATES = [
    0.01,
    0.001,
    0.0001,
]

BATCH_SIZES = [
    32,
    64,
]

DROPOUT_RATES = [
    0.3,
    0.5,
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
    / "part4"
)

MODEL_FOLDER = (
    PROJECT_FOLDER
    / "models"
    / "Homework3"
)


def load_image(filepath, label):
    """
    Load, resize, and normalize one image.
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


def create_dataset(
    csv_filename,
    batch_size,
    training=False,
):
    """
    Create a TensorFlow dataset from a Part 2 CSV file.
    """

    dataframe = pd.read_csv(
        SPLIT_FOLDER / csv_filename
    )

    # Allow both absolute and project-relative filepaths
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

    dataset = dataset.batch(
        batch_size
    )

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal"
                ),

                tf.keras.layers.RandomRotation(
                    0.05
                ),

                tf.keras.layers.RandomBrightness(
                    0.15,
                    value_range=(0.0, 1.0),
                ),
            ]
        )

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

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def build_model(
    learning_rate,
    dropout_rate,
):
    """
    Create the CNN using the selected settings.
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

            tf.keras.layers.Dropout(
                dropout_rate
            ),

            tf.keras.layers.Dense(
                len(CLASS_NAMES),
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    """
    Test every hyperparameter combination.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_experiments = (
        len(LEARNING_RATES)
        * len(BATCH_SIZES)
        * len(DROPOUT_RATES)
    )

    experiment_number = 1
    results = []

    best_validation_loss = float("inf")
    best_settings = None

    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for dropout_rate in DROPOUT_RATES:

                print("\n" + "=" * 60)

                print(
                    f"Experiment "
                    f"{experiment_number}/"
                    f"{total_experiments}"
                )

                print(
                    f"Learning rate: "
                    f"{learning_rate}"
                )

                print(
                    f"Batch size: "
                    f"{batch_size}"
                )

                print(
                    f"Dropout rate: "
                    f"{dropout_rate}"
                )

                print("=" * 60)

                # Clear the previous model from memory
                tf.keras.backend.clear_session()

                tf.keras.utils.set_random_seed(
                    SEED
                )

                train_dataset = create_dataset(
                    "train_split.csv",
                    batch_size,
                    training=True,
                )

                validation_dataset = create_dataset(
                    "validation_split.csv",
                    batch_size,
                )

                model = build_model(
                    learning_rate,
                    dropout_rate,
                )

                early_stopping = (
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=2,
                        restore_best_weights=True,
                    )
                )

                history = model.fit(
                    train_dataset,
                    validation_data=validation_dataset,
                    epochs=EPOCHS,
                    callbacks=[early_stopping],
                    verbose=2,
                )

                validation_loss, validation_accuracy = (
                    model.evaluate(
                        validation_dataset,
                        verbose=0,
                    )
                )

                epochs_completed = len(
                    history.history["loss"]
                )

                results.append(
                    {
                        "experiment": experiment_number,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "dropout_rate": dropout_rate,
                        "epochs_completed": epochs_completed,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                    }
                )

                print(
                    f"Validation loss: "
                    f"{validation_loss:.4f}"
                )

                print(
                    f"Validation accuracy: "
                    f"{validation_accuracy:.4f}"
                )

                # Save the model whenever a new best
                # validation loss is found
                if (
                    validation_loss
                    < best_validation_loss
                ):
                    best_validation_loss = (
                        validation_loss
                    )

                    best_settings = {
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "dropout_rate": dropout_rate,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                    }

                    model.save(
                        MODEL_FOLDER
                        / "best_tuned_model.keras"
                    )

                    model.save_weights(
                        MODEL_FOLDER
                        / "best_tuned_model.weights.h5"
                    )

                    print(
                        "New best model saved."
                    )

                experiment_number += 1

    # Save all experiment results
    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe.sort_values(
            "validation_loss"
        )
    )

    results_dataframe.to_csv(
        OUTPUT_FOLDER
        / "hyperparameter_results.csv",
        index=False,
    )

    # Evaluate the best model once on the test set
    best_model = tf.keras.models.load_model(
        MODEL_FOLDER
        / "best_tuned_model.keras"
    )

    test_dataset = create_dataset(
        "test_split.csv",
        best_settings["batch_size"],
    )

    test_loss, test_accuracy = (
        best_model.evaluate(
            test_dataset,
            verbose=0,
        )
    )

    best_settings["test_loss"] = test_loss
    best_settings["test_accuracy"] = test_accuracy

    # Save the best configuration
    with (
        OUTPUT_FOLDER
        / "best_configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_settings,
            file,
            indent=4,
        )

    results_text = (
        "Best Hyperparameter Configuration\n"
        "---------------------------------\n"
        f"Learning rate: "
        f"{best_settings['learning_rate']}\n"
        f"Batch size: "
        f"{best_settings['batch_size']}\n"
        f"Dropout rate: "
        f"{best_settings['dropout_rate']}\n"
        f"Validation loss: "
        f"{best_settings['validation_loss']:.4f}\n"
        f"Validation accuracy: "
        f"{best_settings['validation_accuracy']:.4f}\n"
        f"Test loss: "
        f"{test_loss:.4f}\n"
        f"Test accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    (
        OUTPUT_FOLDER
        / "best_tuned_results.txt"
    ).write_text(
        results_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(results_text)
    print("Hyperparameter tuning complete.")


if __name__ == "__main__":
    main()