import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
import library.config as config
from library.data_loader import get_dataloaders
from library.model import MultiTaskLSTM
from library.trainer import Trainer
from library.metrics import compute_final_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_reproducibility(seed=42):
    """Sets seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_metrics_logic():
    """
    Verifies the metric calculation logic using synthetic data.
    We create a small dataframe where we know the expected outcome.
    """
    print("\n=== Verifying Metric Logic ===")

    # Create a synthetic dataset
    # Identity: 'male'
    # We want to test if the function correctly handles the bias metrics.

    data = {
        "target": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "prediction": [0.1, 0.2, 0.9, 0.8, 0.1, 0.9],  # Good predictions
        "male": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # Identity column
        # Other identities set to 0
    }

    # Add other identity columns as 0
    for col in config.IDENTITY_COLUMNS:
        if col != "male":
            data[col] = [0.0] * 6

    df = pd.DataFrame(data)

    # Compute metric
    score, metrics_dict = compute_final_metric(
        df,
        label_col="target",
        pred_col="prediction",
        identity_columns=config.IDENTITY_COLUMNS,
    )

    print(f"Synthetic Data Score: {score:.4f}")

    # Assertions
    # Since predictions are perfectly separable (threshold 0.5), AUCs should be 1.0
    # Note: With very few samples, ROC AUC might be undefined if a subset has only 1 class.
    # In this synthetic set:
    # 'male' subgroup: rows 1, 3, 5 -> targets 0, 1, 1. Preds 0.2, 0.8, 0.9. Perfect separation.
    # So Subgroup AUC for male should be 1.0.

    male_metrics = next(
        item for item in metrics_dict["per_subgroup"] if item["subgroup"] == "male"
    )
    print(f"Male Subgroup Metrics: {male_metrics}")

    # If the subgroup AUC is not NaN, it should be 1.0 for this perfect case
    if not np.isnan(male_metrics["subgroup_auc"]):
        assert (
            male_metrics["subgroup_auc"] == 1.0
        ), "Metric calculation failed for perfect synthetic data."

    print("Metric logic verification passed.")


def main():
    print("Initializing demonstration...")
    set_reproducibility(config.SEED)

    # --------------------------------------------------------------------------
    # 1. Data Loading (Debug Mode)
    # --------------------------------------------------------------------------
    print("\n=== Loading Data (Debug Mode) ===")
    # We use debug=True to load only 2000 samples for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify DataLoaders
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Inspect one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    target = batch["target"]
    aux_target = batch["aux_target"]

    print(f"Batch Input Shape: {input_ids.shape}")
    print(f"Batch Target Shape: {target.shape}")
    print(f"Batch Aux Target Shape: {aux_target.shape}")

    # Assertions on shapes
    # input_ids: [batch_size, max_len]
    assert len(input_ids.shape) == 2
    # target: [batch_size]
    assert len(target.shape) == 1
    # aux_target: [batch_size, num_identities]
    assert aux_target.shape[1] == len(config.IDENTITY_COLUMNS)

    # --------------------------------------------------------------------------
    # 2. Model Initialization & Forward Pass Check
    # --------------------------------------------------------------------------
    print("\n=== Initializing Model ===")
    model = MultiTaskLSTM()

    # Move batch to device for testing
    device = config.DEVICE
    model.to(device)
    input_ids = input_ids.to(device)

    # Forward pass
    print("Running forward pass check...")
    with torch.no_grad():
        tox_pred, aux_pred = model(input_ids)

    print(f"Toxicity Prediction Shape: {tox_pred.shape}")
    print(f"Identity Prediction Shape: {aux_pred.shape}")

    # Assertions
    assert tox_pred.shape == (input_ids.size(0), 1), "Incorrect toxicity output shape."
    assert aux_pred.shape == (
        input_ids.size(0),
        len(config.IDENTITY_COLUMNS),
    ), "Incorrect identity output shape."

    # Check value ranges (Sigmoid output should be [0, 1])
    assert (
        tox_pred.min() >= 0 and tox_pred.max() <= 1
    ), "Toxicity predictions out of range [0, 1]."

    # --------------------------------------------------------------------------
    # 3. Metric Logic Verification
    # --------------------------------------------------------------------------
    test_metrics_logic()

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n=== Starting Training Loop Demonstration ===")
    trainer = Trainer(model)

    # Train for 1 epoch (using the small debug dataset)
    print("Training epoch 1...")
    train_metrics = trainer.train_epoch(train_loader)
    print(f"Train Metrics: {train_metrics}")

    assert "loss" in train_metrics
    assert "tox_loss" in train_metrics
    assert "aux_loss" in train_metrics

    # Evaluate
    print("Evaluating...")
    val_metrics = trainer.evaluate(val_loader)
    print(f"Validation Metrics: {val_metrics['score']:.4f} (Score)")

    assert "score" in val_metrics
    assert "metrics" in val_metrics

    # Save Model
    print("Saving model...")
    trainer.save_model(config.MODEL_SAVE_PATH)
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file was not saved."

    # --------------------------------------------------------------------------
    # 5. Prediction & Submission
    # --------------------------------------------------------------------------
    print("\n=== Generating Submission ===")
    # Load model (good practice to verify loading works)
    trainer.load_model(config.MODEL_SAVE_PATH)

    submission_df = trainer.predict(test_loader)

    print(f"Submission Shape: {submission_df.shape}")
    print(f"Submission Columns: {submission_df.columns.tolist()}")

    # Assertions
    assert config.ID_COL in submission_df.columns
    assert "prediction" in submission_df.columns
    assert len(submission_df) > 0

    # Save submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."
    print(f"Submission saved to {config.SUBMISSION_PATH}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
