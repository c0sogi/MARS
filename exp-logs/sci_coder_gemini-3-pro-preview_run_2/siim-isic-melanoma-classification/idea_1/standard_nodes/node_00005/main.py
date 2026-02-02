import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Ensure the library module can be found
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HybridLinearProbe
from library.engine import train_one_epoch, evaluate, predict


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates error magnitude and correlates it with metadata features.
    """
    print("\n[Failure Analysis]")
    model.eval()
    all_targets = []
    all_probs = []

    # Collect predictions
    with torch.no_grad():
        for images, meta, targets in val_loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Error Magnitude (Absolute Error)
    # For binary classification, this is |y - p|
    errors = np.abs(all_targets - all_probs)

    # Retrieve Metadata DataFrame
    # Note: val_loader is not shuffled, so order is preserved
    df_val = val_loader.dataset.df.copy()

    # Add error to dataframe
    df_val["error_magnitude"] = errors

    # Features to analyze
    features = ["age_approx", "sex", "anatom_site_general_challenge"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        if feat not in df_val.columns:
            continue

        # Handle missing values for analysis
        if df_val[feat].dtype == "object":
            # Fill NaNs with a placeholder
            series = df_val[feat].fillna("Missing")
            # Factorize to convert to numeric codes
            series_encoded, _ = pd.factorize(series)
        else:
            # Fill numerical NaNs with mean
            series_encoded = df_val[feat].fillna(df_val[feat].mean())

        # Compute correlation
        if len(np.unique(series_encoded)) > 1:
            corr = np.corrcoef(series_encoded, errors)[0, 1]
            print(f"  {feat}: {corr:.10f}")
        else:
            print(f"  {feat}: N/A (Constant value)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading DataLoaders...")
    # load_cached_data=True allows using pre-computed metadata arrays if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine metadata dimension from a sample
    sample_img, sample_meta, _ = next(iter(train_loader))
    meta_dim = sample_meta.shape[1]
    print(f"Metadata feature dimension: {meta_dim}")

    # 3. Model Initialization
    print("Initializing HybridLinearProbe model...")
    model = HybridLinearProbe(meta_dim=meta_dim)
    model = model.to(device)

    # 4. Training Configuration
    # Using pos_weight to handle class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    patience = 3
    epochs_no_improve = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Cite solution_lesson_node_00004: Strictly employ Early Stopping and Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training completed. Best Validation AUC: {best_auc:.6f}")

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, final_auc = evaluate(model, val_loader, criterion, device)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission Generation
    if final_auc > 0.8378491615214976:
        print("Generating predictions for test set...")
        preds = predict(model, test_loader, device)

        image_names = test_loader.dataset.df["image_name"].values
        submission_df = pd.DataFrame({"image_name": image_names, "target": preds})

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation Metric {final_auc} did not meet threshold 0.8378491615214976. Skipping submission."
        )


if __name__ == "__main__":
    main()
