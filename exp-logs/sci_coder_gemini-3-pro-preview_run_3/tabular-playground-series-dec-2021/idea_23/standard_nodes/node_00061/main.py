import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.model import DeepParallelDCNResNet
from library.train import Trainer, set_seed
from library.data_utils import get_datasets

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_feature_names():
    """
    Reconstructs feature names based on logic in library/data_utils.py
    for meaningful failure analysis.
    """
    # Load a small sample of raw data to get columns
    df = pd.read_parquet(Config.TRAIN_DATA_PATH).iloc[:1]

    # Apply feature engineering to get new columns
    # Re-implementing small parts of logic solely for name extraction
    # to avoid modifying library files.
    df["Aspect_Sin"] = 0
    df["Aspect_Cos"] = 0
    df["Euclidean_Distance_To_Hydrology"] = 0
    df["Absolute_Hydrology_Elevation"] = 0
    df["Mean_Distance_To_Amenities"] = 0

    # Drop ID and Target
    drop_cols = [Config.ID_COL, Config.TARGET_COL]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Define groups as per data_utils.py
    base_cont_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]
    new_cont_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Absolute_Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]

    all_cont_cols = [c for c in base_cont_cols + new_cont_cols if c in df.columns]
    bin_cols = [c for c in df.columns if c not in all_cont_cols]

    # Final order is continuous then binary
    return all_cont_cols + bin_cols


def main():
    # 1. Setup and Configuration
    # Override EPOCHS for fast baseline execution
    Config.EPOCHS = 15
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset, test_ids, classes = get_datasets(
        load_cached_data=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(classes)

    print(f"Input Dimension: {input_dim}, Num Classes: {num_classes}")

    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Training
    trainer = Trainer(model, device, Config)
    trainer.fit(train_loader, val_loader)

    # 5. Final Validation
    print("Performing final validation on best model...")
    val_loss, val_acc = trainer.evaluate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("\nStarting Failure Analysis...")
    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Error vector: 1 if incorrect, 0 if correct
    errors = (preds != targets).astype(int)
    error_rate = errors.mean()
    print(f"Overall Error Rate: {error_rate:.6f}")

    # Correlation Analysis
    # Access validation features from dataset (CPU tensor)
    X_val_tensor = val_dataset.tensors[0]
    X_val_np = X_val_tensor.numpy()

    # Calculate correlation between each feature and the error vector
    correlations = []
    for i in range(X_val_np.shape[1]):
        feat_col = X_val_np[:, i]
        # Handle constant columns (std=0) to avoid NaN
        if np.std(feat_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append(corr)

    # Map to feature names
    try:
        feature_names = get_feature_names()
        if len(feature_names) != len(correlations):
            feature_names = [f"Feature_{i}" for i in range(len(correlations))]
    except Exception as e:
        print(f"Could not reconstruct feature names: {e}")
        feature_names = [f"Feature_{i}" for i in range(len(correlations))]

    corr_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Correlation": correlations,
            "AbsCorrelation": np.abs(correlations),
        }
    )

    print(
        "\nTop 5 Features correlated with Error (Positive = High feature value implies Error):"
    )
    print(
        corr_df.sort_values("Correlation", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    print(
        "\nTop 5 Features negatively correlated with Error (Low feature value implies Error):"
    )
    print(
        corr_df.sort_values("Correlation", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    # 7. Submission Logic
    THRESHOLD = 0.9625041666666667

    if val_acc > THRESHOLD:
        print(
            f"\nValidation Accuracy ({val_acc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict using trainer utility
        raw_preds = trainer.predict(test_loader)
        final_preds = classes[raw_preds]

        # Save
        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
        )
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Accuracy ({val_acc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
