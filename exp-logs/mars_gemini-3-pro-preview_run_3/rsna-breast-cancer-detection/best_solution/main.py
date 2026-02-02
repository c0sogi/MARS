import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_dataloaders
from library.trainer import Trainer
from library.inference import run_inference
from library.utils import seed_everything


def calculate_pf1(y_true, y_pred):
    """
    Calculates the Probabilistic F1 score (pF1).

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Where:
    pTP = sum(y_pred * y_true)
    pFP = sum(y_pred * (1 - y_true))
    TP + FN = sum(y_true)
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid numerical instability
    epsilon = 1e-7

    # Calculate terms
    pTP = np.sum(y_pred * y_true)
    pFP = np.sum(y_pred * (1 - y_true))
    total_positive = np.sum(y_true)

    # Calculate Precision and Recall
    # pPrecision = pTP / (Sum of predicted probabilities)
    pPrecision = pTP / (pTP + pFP + epsilon)

    # pRecall = pTP / (Total Positives)
    pRecall = pTP / (total_positive + epsilon)

    # Calculate F1
    pf1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall + epsilon)

    return pf1


def perform_failure_analysis(y_true, y_pred):
    """
    Correlates model error with input features on the validation set.
    """
    print("\n=== Failure Analysis ===")

    # Load metadata
    try:
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    except FileNotFoundError:
        print("Validation metadata not found. Skipping failure analysis.")
        return

    # Ensure lengths match (Data loader should preserve order if shuffle=False)
    if len(val_df) != len(y_pred):
        print(
            f"Warning: Metadata length ({len(val_df)}) does not match prediction length ({len(y_pred)}). Skipping analysis."
        )
        return

    # Calculate Error Magnitude
    val_df["error"] = np.abs(y_true - y_pred)

    # Preprocess features for correlation
    # Map Density: A->1, B->2, C->3, D->4
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    val_df["density_encoded"] = val_df["density"].map(density_map)

    # Map Laterality: L->0, R->1
    val_df["laterality_encoded"] = val_df["laterality"].map({"L": 0, "R": 1})

    # Map View: CC->0, MLO->1 (Others ignored or mapped to NaN)
    val_df["view_encoded"] = val_df["view"].map({"CC": 0, "MLO": 1})

    # Select features to correlate
    features = [
        "age",
        "density_encoded",
        "laterality_encoded",
        "view_encoded",
        "implant",
        "machine_id",
    ]

    correlations = {}
    for feat in features:
        if feat in val_df.columns:
            # Drop NaNs for calculation
            valid_data = val_df[[feat, "error"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feat].corr(valid_data["error"])
                correlations[feat] = corr

    print("Correlation between Error Magnitude and Features:")
    for feat, corr in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {feat}: {corr:.4f}")


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Initializing pipeline...")

    # 2. Data Loading
    dataloaders = get_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 3. Training
    # Initialize Trainer
    trainer = Trainer()

    # Fast Baseline Config: 5 Epochs, Max 150 batches per epoch
    # This ensures quick execution while seeing enough data.
    print("Starting training (Fast Baseline Mode)...")
    trainer.fit(train_loader, val_loader, epochs=5, max_batches=150)

    # 4. Validation & Metric Calculation
    print("\nRunning validation inference for metric calculation...")

    # Load best model weights
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=Config.DEVICE)
        )

    trainer.model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(images)
            probs = outputs.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(labels.numpy().flatten())

    # Calculate Metric
    pf1_score = calculate_pf1(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {pf1_score}")

    # 5. Failure Analysis
    perform_failure_analysis(np.array(all_targets), np.array(all_preds))

    # 6. Submission
    print("\nGenerating submission file...")
    # run_inference handles the test loader and submission generation internally
    run_inference(output_path=Config.SUBMISSION_PATH)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run()
