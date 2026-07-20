from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# general settings
SEED = 42
IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Beta",
    "Cray",
    "Discuss",
    "Gold",
    "Guppy",
    "Oscar",
]


# Project folders
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

SPLIT_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part2"
)

PART3_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part3"
)

PART4_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part4"
)

OUTPUT_FOLDER = (
    PROJECT_FOLDER
    / "images"
    / "output"
    / "Homework3"
    / "part5"
)

MODEL_FOLDER = (
    PROJECT_FOLDER
    / "models"
    / "Homework3"
)


# augmentation data used during training
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


def create_dataset(
    csv_filename,
    batch_size=32,
    training=False,
):
    """
    dataset from a split CSV file.
    """

    dataframe = pd.read_csv(
        SPLIT_FOLDER / csv_filename
    )

    # Handle both absolute and project-relative paths
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

    return dataset, dataframe


def build_model(
    learning_rate=0.001,
    dropout_rate=0.5,
):
    """
    Recreate the CNN architecture used in Parts 3 and 4
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


def predict_classes(model, dataset):
    """
    Return the predicted class number for each image.
    """

    probabilities = model.predict(
        dataset,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return predictions


def evaluate_predictions(
    model_name,
    true_labels,
    predicted_labels,
):
    """
    print and save the classification report
    """

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(f"\n{model_name} Classification Report")
    print("-" * 60)
    print(report_text)

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report_dictionary
    ).transpose()

    report_dataframe.to_csv(
        OUTPUT_FOLDER
        / f"{model_name.lower()}_classification_report.csv"
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def recreate_optimized_history(
    learning_rate,
    batch_size,
    dropout_rate,
):
    """
    Retrain the winning configuration once so its
    training curves can be created
    """

    print(
        "\nRecreating optimized training history..."
    )

    train_dataset, _ = create_dataset(
        "train_split.csv",
        batch_size=batch_size,
        training=True,
    )

    validation_dataset, _ = create_dataset(
        "validation_split.csv",
        batch_size=batch_size,
        training=False,
    )

    tf.keras.utils.set_random_seed(
        SEED
    )

    model = build_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
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
        epochs=8,
        callbacks=[early_stopping],
        verbose=2,
    )

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe.to_csv(
        OUTPUT_FOLDER
        / "optimized_training_history.csv",
        index=False,
    )

    return history_dataframe


def create_evaluation_grid(
    baseline_history,
    optimized_history,
    optimized_matrix,
):
    """
    Create the image grid.
    """

    figure = plt.figure(
        figsize=(18, 10)
    )

    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 1.2],
    )

    # Baseline accuracy
    axis1 = figure.add_subplot(
        grid[0, 0]
    )

    axis1.plot(
        baseline_history["accuracy"],
        label="Training",
    )

    axis1.plot(
        baseline_history["val_accuracy"],
        label="Validation",
    )

    axis1.set_title(
        "Baseline Accuracy"
    )

    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("Accuracy")
    axis1.legend()

    # Optimized accuracy
    axis2 = figure.add_subplot(
        grid[0, 1]
    )

    axis2.plot(
        optimized_history["accuracy"],
        label="Training",
    )

    axis2.plot(
        optimized_history["val_accuracy"],
        label="Validation",
    )

    axis2.set_title(
        "Optimized Accuracy"
    )

    axis2.set_xlabel("Epoch")
    axis2.set_ylabel("Accuracy")
    axis2.legend()

    # Baseline loss
    axis3 = figure.add_subplot(
        grid[1, 0]
    )

    axis3.plot(
        baseline_history["loss"],
        label="Training",
    )

    axis3.plot(
        baseline_history["val_loss"],
        label="Validation",
    )

    axis3.set_title(
        "Baseline Loss"
    )

    axis3.set_xlabel("Epoch")
    axis3.set_ylabel("Loss")
    axis3.legend()

    # optimized loss
    axis4 = figure.add_subplot(
        grid[1, 1]
    )

    axis4.plot(
        optimized_history["loss"],
        label="Training",
    )

    axis4.plot(
        optimized_history["val_loss"],
        label="Validation",
    )

    axis4.set_title(
        "Optimized Loss"
    )

    axis4.set_xlabel("Epoch")
    axis4.set_ylabel("Loss")
    axis4.legend()

    #Optimized confusion matrix
    axis5 = figure.add_subplot(
        grid[:, 2]
    )

    matrix_image = axis5.imshow(
        optimized_matrix,
        cmap="Blues",
    )

    axis5.set_title(
        "Optimized Model Confusion Matrix"
    )

    axis5.set_xlabel(
        "Predicted Class"
    )

    axis5.set_ylabel(
        "Actual Class"
    )

    axis5.set_xticks(
        range(len(CLASS_NAMES))
    )

    axis5.set_yticks(
        range(len(CLASS_NAMES))
    )

    axis5.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis5.set_yticklabels(
        CLASS_NAMES
    )

    for row in range(
        len(CLASS_NAMES)
    ):
        for column in range(
            len(CLASS_NAMES)
        ):
            axis5.text(
                column,
                row,
                optimized_matrix[row, column],
                ha="center",
                va="center",
            )

    figure.colorbar(
        matrix_image,
        ax=axis5,
        fraction=0.046,
    )

    figure.suptitle(
        "Homework Three Model Evaluation",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FOLDER
        / "model_evaluation_grid.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """
    Evaluate and compare both models.
    """

    tf.keras.utils.set_random_seed(
        SEED
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Part 5: Evaluation and Analysis")
    print("=" * 60)

    # load the held-out test set
    test_dataset, test_dataframe = (
        create_dataset(
            "test_split.csv",
            batch_size=32,
            training=False,
        )
    )

    true_labels = (
        test_dataframe["label"]
        .astype("int32")
        .to_numpy()
    )

    # Rebuild and load the baseline model
    print("\nLoading baseline model...")

    baseline_model = build_model(
        learning_rate=0.001,
        dropout_rate=0.5,
    )

    baseline_model.load_weights(
        MODEL_FOLDER
        / "baseline_cnn.weights.h5"
    )

    # Load the optimized model
    print("Loading optimized model...")

    optimized_model = (
        tf.keras.models.load_model(
            MODEL_FOLDER
            / "best_tuned_model.keras"
        )
    )

    #generate predictions
    baseline_predictions = predict_classes(
        baseline_model,
        test_dataset,
    )

    optimized_predictions = predict_classes(
        optimized_model,
        test_dataset,
    )

    # Create classification reports
    baseline_metrics = evaluate_predictions(
        "Baseline",
        true_labels,
        baseline_predictions,
    )

    optimized_metrics = evaluate_predictions(
        "Optimized",
        true_labels,
        optimized_predictions,
    )

    # Save overall metric comparison,
    metric_comparison = pd.DataFrame(
        [
            baseline_metrics,
            optimized_metrics,
        ]
    )

    metric_comparison.to_csv(
        OUTPUT_FOLDER
        / "model_metric_comparison.csv",
        index=False,
    )

    print("\nOverall Model Comparison")
    print("-" * 60)

    print(
        metric_comparison.to_string(
            index=False
        )
    )

    # Create and save the optimized confusion matrix
    optimized_matrix = confusion_matrix(
        true_labels,
        optimized_predictions,
        labels=range(len(CLASS_NAMES)),
    )

    pd.DataFrame(
        optimized_matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        OUTPUT_FOLDER
        / "optimized_confusion_matrix.csv"
    )

    # Load baseline history
    baseline_history = pd.read_csv(
        PART3_FOLDER
        / "baseline_training_history.csv"
    )

    # Read winning Part 4 settings
    with (
        PART4_FOLDER
        / "best_configuration.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        best_settings = json.load(
            file
        )

    #recreate only the optimized model's curves
    optimized_history = (
        recreate_optimized_history(
            learning_rate=best_settings[
                "learning_rate"
            ],
            batch_size=best_settings[
                "batch_size"
            ],
            dropout_rate=best_settings[
                "dropout_rate"
            ],
        )
    )

    create_evaluation_grid(
        baseline_history,
        optimized_history,
        optimized_matrix,
    )

    # Save a readable summary
    qualitative_analysis = (
        "Qualitative Analysis\n"
        "--------------------\n"
        "Random horizontal flipping helped the model learn that "
        "fish direction should not determine the class. Minor "
        "rotations reduced dependence on exact image orientation. "
        "Brightness adjustment helped the model handle lighting "
        "differences. These transformations also added variation to the "
        "training set, although they also caused some normal "
        "fluctuation in validation loss.\n\n"
        "The best tuning configuration used a learning rate of "
        f"{best_settings['learning_rate']}, a batch size of "
        f"{best_settings['batch_size']}, and a dropout rate of "
        f"{best_settings['dropout_rate']}. The learning rate of "
        "0.001 provided the strongest balance between fast learning "
        "and stable validation results. A dropout rate of 0.5 helped "
        "reduce overfitting more than 0.3 in the best-performing "
        "experiments. The batch size of 32 also performed better "
        "than 64 for the winning configuration.\n\n"
        "The optimized model did not outperform the baseline on the "
        "held-out test set. This shows that selecting a model by the "
        "lowest validation loss does not guarantee a higher final "
        "test accuracy.\n"
    )

    summary_text = (
        qualitative_analysis
        + "\nQuantitative Comparison\n"
        + "-----------------------\n"
        + f"Baseline accuracy: "
        f"{baseline_metrics['accuracy']:.4f}\n"
        + f"Baseline precision: "
        f"{baseline_metrics['precision']:.4f}\n"
        + f"Baseline recall: "
        f"{baseline_metrics['recall']:.4f}\n"
        + f"Baseline F1-score: "
        f"{baseline_metrics['f1_score']:.4f}\n\n"
        + f"Optimized accuracy: "
        f"{optimized_metrics['accuracy']:.4f}\n"
        + f"Optimized precision: "
        f"{optimized_metrics['precision']:.4f}\n"
        + f"Optimized recall: "
        f"{optimized_metrics['recall']:.4f}\n"
        + f"Optimized F1-score: "
        f"{optimized_metrics['f1_score']:.4f}\n"
    )

    (
        OUTPUT_FOLDER
        / "part5_evaluation_summary.txt"
    ).write_text(
        summary_text,
        encoding="utf-8",
    )

    print("\n" + summary_text)

    print(
        "\nPart 5 files saved in:"
    )

    print(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()