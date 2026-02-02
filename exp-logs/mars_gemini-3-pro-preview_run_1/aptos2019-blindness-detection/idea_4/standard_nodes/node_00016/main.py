import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Import provided library modules
from library.utils import (
    seed_everything,
    quadratic_weighted_kappa,
    decode_ordinal_predictions,
)
from library.dataset import create_dataloaders
from library.trainer import DRTrainer
from library.model import OrdinalModel


def get_image_meta_features(file_rel_path, input_dir):
    """
    Extracts meta-features (width, height, aspect_ratio, file_size, mean_intensity)
    for a single image file. Returns None if file is missing/corrupt.
    """
    full_path = os.path.join(input_dir, file_rel_path)
    if not os.path.exists(full_path):
        return None

    try:
        # File size
        file_size = os.path.getsize(full_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        h, w, c = img.shape
        mean_intensity = img.mean()
        aspect_ratio = w / h if h > 0 else 0

        return [w, h, aspect_ratio, file_size, mean_intensity]
    except Exception:
        return None


def main():
    # 1. Configuration
    SEED = 42
    seed_everything(SEED)

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/experiment"
    SUBMISSION_PATH = "./submission/submission.csv"

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Hyperparameters
    # Cite Lesson 00004: EfficientNet-B0 + 256x256 offered best trade-off (0.92 QWK)
    # Cite Lesson 00007: Prioritize batch size stability (32) over resolution (512)
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 20  # Increased to ensure convergence with smaller model
    LR = 1e-3  # Cite Lesson 00006: Scale LR for larger batch size
    NUM_WORKERS = 4

    # Threshold for submission
    SUBMISSION_THRESHOLD = 0.9194950903896975

    # 2. Training
    print(f"Initializing Trainer for {MODEL_NAME} @ {IMG_SIZE}x{IMG_SIZE}...")
    trainer = DRTrainer(
        experiment_dir=WORKING_DIR,
        model_name=MODEL_NAME,
        img_size=IMG_SIZE,
        num_classes=4,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Starting training loop...")
    # fit returns the best validation QWK score achieved
    best_val_score = trainer.fit(train_csv=TRAIN_CSV, val_csv=VAL_CSV)

    # 3. Validation & Failure Analysis
    print("\n=== Validation & Failure Analysis ===")

    # Load the best model explicitly for analysis
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()

    # Create validation dataloader
    _, val_loader, _ = create_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
        seed=SEED,
    )

    # Inference on validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(trainer.device)
            outputs = trainer.model(images)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)

    # Decode
    y_pred = decode_ordinal_predictions(preds_tensor)
    y_true = targets_tensor.sum(dim=1).numpy().astype(int)

    # Compute and Print Final Metric
    final_metric = quadratic_weighted_kappa(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with meta-features
    print("Calculating error correlations...")
    df_val = pd.read_csv(VAL_CSV)

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Extract meta features for each validation image
    meta_features_list = []
    valid_indices = []

    # Iterate through dataframe to maintain alignment
    for idx, row in df_val.iterrows():
        feats = get_image_meta_features(row["file_path"], INPUT_DIR)
        if feats is not None:
            meta_features_list.append(feats)
            valid_indices.append(idx)
        else:
            # If image read fails, we skip it in correlation analysis
            pass

    if len(meta_features_list) > 0:
        meta_features = np.array(meta_features_list)
        relevant_errors = errors[valid_indices]

        feature_names = [
            "width",
            "height",
            "aspect_ratio",
            "file_size",
            "mean_intensity",
        ]

        print("\nCorrelation between Error Magnitude and Input Features:")
        for i, name in enumerate(feature_names):
            # Check for constant values to avoid warning
            feat_col = meta_features[:, i]
            if np.std(feat_col) == 0 or np.std(relevant_errors) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(feat_col, relevant_errors)
            print(f"{name}: {corr:.4f}")
    else:
        print("Could not extract meta-features for failure analysis.")

    # 4. Submission
    print("\n=== Submission Check ===")
    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        trainer.predict_and_submit(TEST_CSV, submission_path=SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
