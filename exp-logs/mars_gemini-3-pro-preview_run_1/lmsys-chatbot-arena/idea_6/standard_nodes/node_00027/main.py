import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import get_dataloaders
from library.model import SiameseDebertaGeometric
from library.trainer import Trainer


def main():
    # 1. Setup Environment
    # Override Config for fast baseline execution
    # Cite solution_lesson_node_00015: Increase epochs to ensure convergence (Val Loss < Train Loss)
    Config.NUM_EPOCHS = 2
    Config.setup_environment()
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Running on device: {device}")
    print(
        f"Training for {Config.NUM_EPOCHS} epoch(s) with batch size {Config.TRAIN_BATCH_SIZE} (accum={Config.ACCUMULATION_STEPS})."
    )

    # 2. Data Loading
    # We use the full dataset (debug=False) but limited epochs to ensure we hit the performance threshold
    # while keeping runtime low.
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SiameseDebertaGeometric()
    model.to(device)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # 5. Validation & Metric Calculation
    print("\nLoading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_labels = []
    val_scalars = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )
            probs = torch.softmax(logits, dim=1)

            val_preds.append(probs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
            val_scalars.append(scalar_features.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_scalars = np.concatenate(val_scalars, axis=0)

    # Compute Log Loss
    # Ensure float64 for precision
    final_metric = log_loss(val_labels.astype(np.float64), val_preds.astype(np.float64))
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    eps = 1e-15
    preds_clipped = np.clip(val_preds, eps, 1 - eps)
    # Cross entropy per sample: -sum(y_true * log(y_pred))
    sample_losses = -np.sum(val_labels * np.log(preds_clipped), axis=1)

    feature_names = [
        "diff_char",
        "diff_word",
        "diff_newline",
        "ratio_char",
        "ratio_word",
        "ratio_newline",
    ]
    print("Correlation between Error Magnitude (Log Loss) and Scalar Features:")

    for i, name in enumerate(feature_names):
        feat_values = val_scalars[:, i]
        # Check for valid values
        mask = np.isfinite(feat_values) & np.isfinite(sample_losses)
        if mask.sum() > 1:
            corr = np.corrcoef(feat_values[mask], sample_losses[mask])[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: NaN (Insufficient data)")

    # 7. Submission
    threshold = 1.0026075514615997
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_preds = []
        print("Running inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)
                scalar_features = batch["scalar_features"].to(device)

                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    scalar_features,
                )
                probs = torch.softmax(logits, dim=1)
                test_preds.append(probs.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Load sample submission to get IDs
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Assign predictions
        sub_df["winner_model_a"] = test_preds[:, 0]
        sub_df["winner_model_b"] = test_preds[:, 1]
        sub_df["winner_tie"] = test_preds[:, 2]

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
