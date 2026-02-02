import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import get_dataloaders
from library.model import TransformerResFunnel
from library.train import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Override Config for this run
    Config.EPOCHS = (
        40  # Increased epochs for convergence (Cite solution_lesson_node_00003)
    )
    Config.SUBMISSION_SAVE_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = TransformerResFunnel().to(device)
    trainer = Trainer(model, device)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_auc = trainer.validate(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # --------------------------------------------------------------------------
    print("Performing final validation assessment...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current state.")

    model.eval()

    # Collect all validation data and predictions for analysis
    val_cont_features = []
    val_targets = []
    val_preds = []

    with torch.no_grad():
        for batch in val_loader:
            x_cont = batch["cont"].to(device)
            x_cat = batch["cat"].to(device)
            y = batch["target"].to(device).unsqueeze(1)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            val_cont_features.append(x_cont.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_preds.append(probs.cpu().numpy())

    val_cont_features = np.concatenate(val_cont_features, axis=0)
    val_targets = np.concatenate(val_targets).flatten()
    val_preds = np.concatenate(val_preds).flatten()

    final_auc = compute_auc(val_targets, val_preds)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between error and continuous features
    print("Correlation between Error Magnitude and Input Features:")
    feature_correlations = []

    for i, feature_name in enumerate(Config.CONT_FEATURES):
        feature_values = val_cont_features[:, i]
        # Compute Pearson correlation
        if np.std(feature_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_values, errors)[0, 1]

        feature_correlations.append((feature_name, corr))

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in feature_correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9967793385748163

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x_cont = batch["cont"].to(device)
                x_cat = batch["cat"].to(device)

                logits = model(x_cont, x_cat)
                probs = torch.sigmoid(logits)
                test_preds.append(probs.cpu().numpy())

        test_preds = np.concatenate(test_preds).flatten()

        # Load test metadata for IDs
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        # Verify length
        if len(test_preds) != len(test_meta):
            print(
                f"Warning: Prediction length {len(test_preds)} != Metadata length {len(test_meta)}"
            )
            # In debug mode, we might need to truncate metadata
            if len(test_preds) < len(test_meta):
                test_meta = test_meta.iloc[: len(test_preds)]

        submission = pd.DataFrame({"id": test_meta["id"], "target": test_preds})

        submission.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
