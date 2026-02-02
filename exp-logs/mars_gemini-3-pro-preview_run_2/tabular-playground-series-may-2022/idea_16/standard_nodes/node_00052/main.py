import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.train import train_model, generate_submission
from library.utils import seed_everything, load_checkpoint, compute_auc
from library.dataset import get_dataloaders
from library.model_components import GatedTransformerResFunnelHybrid


def main():
    # --------------------------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Orchestration started on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # We use 10 epochs and a larger batch size (2048) to speed up execution on A100
    # while using the full dataset to ensure we meet the high AUC threshold.
    print("\n=== Starting Training Phase ===")
    train_model(
        epochs=10,
        batch_size=2048,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        debug=False,
    )

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n=== Starting Validation Phase ===")

    # Load data
    _, val_loader, _, _ = get_dataloaders(batch_size=2048, load_cached_data=True)

    # Initialize model and load best checkpoint
    model = GatedTransformerResFunnelHybrid().to(device)
    load_checkpoint(Config.MODEL_CHECKPOINT, model, device=device)
    model.eval()

    all_targets = []
    all_preds = []
    all_cont_features = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["cont_features"].to(device)
            cat = batch["cat_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_cont_features.append(cont.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    X_cont = np.vstack(all_cont_features)

    # Compute Metric
    val_auc = compute_auc(y_true, y_pred)
    print(f"Final Validation Metric: {val_auc}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Starting Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Reconstruct feature names (f_00 to f_30, excluding f_27)
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    # Calculate correlation between Error and each Continuous Feature
    for i, feat_name in enumerate(feature_names):
        feat_values = X_cont[:, i]
        # Pearson correlation
        if np.std(feat_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_values)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    print("\n=== Checking Submission Criteria ===")
    threshold = 0.9970005855169476

    if val_auc > threshold:
        print(f"Validation AUC ({val_auc:.6f}) > Threshold ({threshold:.6f}).")
        print("Generating submission file...")
        generate_submission(batch_size=2048)
    else:
        print(f"Validation AUC ({val_auc:.6f}) <= Threshold ({threshold:.6f}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
