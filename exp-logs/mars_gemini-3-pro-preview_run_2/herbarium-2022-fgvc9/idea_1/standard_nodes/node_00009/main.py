import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    INPUT_DIR,
)
from library.utils import seed_everything
from library.dataset import PlantDataset, get_transforms
from library.prototype_manager import SupervisedTrainer


def main():
    # 1. Setup & Configuration
    seed_everything(SEED)

    # 2. Data Preparation
    # We use standard ImageNet normalization and resizing
    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    # Create label mapping from training data to ensure contiguous indices
    df_train = pd.read_csv(TRAIN_CSV)
    unique_labels = sorted(df_train["label"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    num_classes = len(unique_labels)
    print(f"Mapped {num_classes} unique classes from training data.")

    # Initialize Datasets with label map
    train_dataset = PlantDataset(
        TRAIN_CSV, transform=train_transforms, label_map=label_map
    )
    val_dataset = PlantDataset(VAL_CSV, transform=val_transforms, label_map=label_map)
    test_dataset = PlantDataset(TEST_CSV, transform=val_transforms, test_mode=True)

    # Initialize DataLoaders
    # pin_memory=True speeds up transfer to GPU
    # IMPORTANT: Shuffle=True for training in supervised learning
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Using SupervisedTrainer which fine-tunes the backbone (Cite solution_lesson_node_00001)
    # Pass dynamic num_classes to prevent device-side assert (Cite debug_lesson_3)
    classifier = SupervisedTrainer(num_classes=num_classes)

    # 4. Training
    classifier.fit(train_loader)

    # 5. Validation
    # Evaluate the model on the hold-out validation set
    f1_score = classifier.evaluate(val_loader)

    # REQUIRED OUTPUT: Print the final validation metric in the specified format
    print(f"Final Validation Metric: {f1_score}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    perform_failure_analysis(classifier, val_loader, val_dataset.df, train_dataset.df)

    # 7. Submission Generation
    if f1_score > 0.6021914648406147:
        classifier.generate_submission(test_loader, idx_to_label=idx_to_label)


def perform_failure_analysis(classifier, val_loader, val_df, train_df):
    """
    Analyzes the model's performance on the validation set to identify error patterns.
    Calculates correlations between error magnitude and input features (class frequency, image dimensions).
    """
    # Get predictions and ground truth labels
    preds, labels = classifier.predict(val_loader, is_test=False)
    preds = np.array(preds)
    labels = np.array(labels)

    # Calculate binary error vector (1 = Incorrect, 0 = Correct)
    errors = (preds != labels).astype(int)

    # --- Analysis 1: Correlation with Class Frequency (Long-tail analysis) ---
    # Count number of training samples per class
    train_counts = train_df["label"].value_counts()

    # Calculate error rate per class in validation set
    # Map labels back to original IDs if dataset has mapping
    dataset = val_loader.dataset
    if hasattr(dataset, "label_map") and dataset.label_map is not None:
        idx_to_label = {v: k for k, v in dataset.label_map.items()}
        original_labels = np.array([idx_to_label[l] for l in labels])
        val_results = pd.DataFrame({"label": original_labels, "error": errors})
    else:
        val_results = pd.DataFrame({"label": labels, "error": errors})

    class_error_rate = val_results.groupby("label")["error"].mean()

    # Create a combined DataFrame aligning counts and error rates by class ID
    analysis_df = pd.DataFrame(
        {"train_samples": train_counts, "val_error_rate": class_error_rate}
    ).dropna()

    if not analysis_df.empty:
        # Calculate Pearson correlation
        corr_freq = analysis_df["train_samples"].corr(analysis_df["val_error_rate"])
        print(
            f"Correlation between Class Training Frequency and Validation Error Rate: {corr_freq:.4f}"
        )
    else:
        print("Insufficient data for class frequency analysis.")

    # --- Analysis 2: Correlation with Image Metadata (Dimensions) ---
    # We analyze a random sample of the validation set to maintain speed
    sample_size = 2000
    if len(val_df) > sample_size:
        rng = np.random.RandomState(SEED)
        sample_indices = rng.choice(len(val_df), sample_size, replace=False)
    else:
        sample_indices = np.arange(len(val_df))

    widths = []
    heights = []
    sampled_errors = []

    # Collect image dimensions for the sampled indices
    for idx in sample_indices:
        row = val_df.iloc[idx]
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        try:
            # Open image lazily to get dimensions without full load
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sampled_errors.append(errors[idx])
        except Exception:
            continue

    if len(sampled_errors) > 10:
        meta_df = pd.DataFrame(
            {
                "width": widths,
                "height": heights,
                "aspect_ratio": np.array(widths) / np.array(heights),
                "error": sampled_errors,
            }
        )

        # Calculate correlations between error (0/1) and image properties
        corr_w = meta_df["width"].corr(meta_df["error"])
        corr_h = meta_df["height"].corr(meta_df["error"])
        corr_ar = meta_df["aspect_ratio"].corr(meta_df["error"])

        print(f"Correlation between Image Width and Error: {corr_w:.4f}")
        print(f"Correlation between Image Height and Error: {corr_h:.4f}")
        print(f"Correlation between Aspect Ratio and Error: {corr_ar:.4f}")
    else:
        print("Insufficient valid images for metadata analysis.")


if __name__ == "__main__":
    main()
