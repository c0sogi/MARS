import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import load_data, IcebergDataset
from library.train import train_model
from library.inference import predict_with_tta, generate_submission
from library.model import IcebergResNet18


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Load Data and Metadata
    # Load full arrays
    print("Loading full dataset...")
    train_images, train_angles, train_labels, _, _, _ = load_data(load_cached_data=True)

    # Load Split Metadata
    print("Loading split metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Extract indices for fixed splits
    # The 'sample_index' in metadata corresponds to the index in the full arrays loaded from train.json
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values

    print(f"Training Size: {len(train_indices)}")
    print(f"Validation Size: {len(val_indices)}")

    # 3. Single Model Training
    # We train on the full training split and validate on the fixed validation split.
    # This maximizes data usage compared to bagging. (Cite solution_lesson_node_00023)
    train_model(train_indices, val_indices, device)

    # 4. Evaluation on Fixed Validation Set
    print("\n=== Evaluating Model on Fixed Validation Set ===")

    # Prepare Validation Data
    X_val = train_images[val_indices]
    a_val = train_angles[val_indices]
    y_val = train_labels[val_indices]

    # Transform (No augmentation, just tensor conversion)
    val_transform = A.Compose([ToTensorV2()])
    val_dataset = IcebergDataset(X_val, a_val, y_val, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    model = IcebergResNet18().to(device)
    model = load_checkpoint(model, model_path, device)

    # Predict with TTA (Cite solution_lesson_node_00005)
    preds = []
    for images, angles, _ in val_loader:
        probs = predict_with_tta(model, images, angles, device)
        preds.append(probs)

    avg_preds = np.concatenate(preds).flatten()

    # Calculate Metric
    final_log_loss = log_loss(y_val, avg_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate Error
    errors = np.abs(y_val - avg_preds)

    # Calculate Image Statistics for Correlation
    # X_val is (N, 224, 224, 3)
    # Channel 0: Band 1, Channel 1: Band 2
    img_means = np.mean(X_val, axis=(1, 2, 3))
    img_stds = np.std(X_val, axis=(1, 2, 3))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": a_val,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Generate Submission
    # Threshold from prompt
    THRESHOLD = 0.21099163245555455

    if final_log_loss < THRESHOLD:
        print(
            f"\nValidation score ({final_log_loss:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation score ({final_log_loss:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
