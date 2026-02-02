import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library files
from library.trainer import Trainer
from library.utils import seed_everything, get_data, _feature_engineering
from library.data_loader import CoverTypeDataset
from library.model import predict_test


def main():
    # 1. Setup and Initialization
    seed_everything(42)

    # Hyperparameters for Fast Baseline
    # Reducing epochs to 15 to satisfy "fast baseline" requirement while ensuring convergence
    EPOCHS = 15
    BATCH_SIZE = 4096

    print("Initializing Trainer...")
    trainer = Trainer(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        warmup_epochs=3,
        patience=5,
        cache_dir="./working/idea_39/",
        metadata_dir="./metadata",
        output_dir="./submission",
    )

    # 2. Train the Model
    # This handles loading data, training, and reloading best weights
    trainer.train()

    # 3. Validation Assessment
    print("\nRunning Validation Assessment...")

    # Load validation data explicitly for analysis
    data = get_data(
        load_cached_data=True, cache_dir="./working/idea_39/", metadata_dir="./metadata"
    )
    X_val = data["val_X"]
    y_val = data["val_y"]  # 0-indexed targets

    # Create DataLoader for Validation
    val_ds = CoverTypeDataset(X_val, y_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Run Inference on Validation Set
    # predict_test returns (ids, preds). When passed a dataset with targets, 'ids' contains the targets.
    true_labels, preds = predict_test(trainer.model, val_loader, trainer.device)

    true_labels = np.array(true_labels)
    preds = np.array(preds)

    # Calculate Final Validation Metric (Accuracy)
    accuracy = np.mean(true_labels == preds)
    print(f"Final Validation Metric: {accuracy}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude (1 if incorrect, 0 if correct)
    error_magnitude = (true_labels != preds).astype(int)

    # Reconstruct feature names to provide meaningful analysis
    # We load the dataframe to get column names and apply the same feature engineering logic
    try:
        # Load full train parquet (efficient with sufficient RAM) to extract schema
        df_train = pd.read_parquet("./metadata/train.parquet")
        df_eng = _feature_engineering(df_train)

        # Replicate the column ordering logic from library.utils.get_data
        target_col = "Cover_Type"
        id_col = "Id"

        # Identify binary columns
        binary_cols = [
            c
            for c in df_eng.columns
            if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
        ]

        # Identify continuous columns
        exclude = [target_col, id_col] + binary_cols
        cont_cols = [c for c in df_eng.columns if c not in exclude]

        # Final feature order matches the concatenation in utils.py
        feature_names = cont_cols + binary_cols

    except Exception as e:
        print(f"Warning: Could not reconstruct feature names ({e}). Using indices.")
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    # Calculate correlation between each feature and the error magnitude
    correlations = []
    for i in range(X_val.shape[1]):
        if i >= len(feature_names):
            break

        feature_values = X_val[:, i]

        # Skip constant features to avoid warnings
        if np.std(feature_values) < 1e-9:
            corr = 0.0
        else:
            # np.corrcoef returns a matrix; we want the off-diagonal element
            corr = np.corrcoef(feature_values, error_magnitude)[0, 1]

        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation to find most impactful features
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Features Correlated with Error Magnitude (Systematic Failure Patterns):")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.6f}")

    # 5. Submission Generation
    THRESHOLD = 0.9626291666666666

    if accuracy > THRESHOLD:
        print(f"\nValidation metric {accuracy} > {THRESHOLD}. Generating submission...")
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
