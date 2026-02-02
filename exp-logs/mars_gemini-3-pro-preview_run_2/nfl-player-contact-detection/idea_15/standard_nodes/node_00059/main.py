import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import joblib
from sklearn.metrics import matthews_corrcoef

# Import from provided library files
from library.config import Config
from library.data_processing import get_data_loaders
from library.model import CA_WRN
from library.trainer import train_one_epoch, validate, FocalLoss
from library.inference import optimize_threshold, generate_submission


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Override Config for Fast Baseline
    # Limit epochs to ensure execution within time limits while allowing convergence
    Config.EPOCHS = 6
    print(f"Configuration: EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}")

    # ------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------
    print("\n=== Loading Data ===")
    # load_cached_data=True will use ./working/idea_15/ cache if available
    train_loader, val_loader, center_indices, scaler = get_data_loaders(
        load_cached_data=True
    )

    # Determine input dimension from a sample
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]
    print(f"Input Dimension: {input_dim}")
    print(f"Center Indices Count: {len(center_indices)}")

    # ------------------------------------------------------------------
    # 3. Model Training
    # ------------------------------------------------------------------
    print("\n=== Initializing Model & Training ===")
    model = CA_WRN(
        input_dim=input_dim,
        center_indices=center_indices,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    best_mcc = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MCC: {val_mcc:.5f}"
        )

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> Saved new best model (MCC: {best_mcc:.5f})")

    print(f"Training finished. Best MCC during training: {best_mcc:.5f}")

    # ------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # ------------------------------------------------------------------
    print("\n=== Final Validation & Threshold Optimization ===")
    # Load best model
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Use the library function to find the best threshold and exact MCC
    best_threshold, final_val_mcc = optimize_threshold(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_mcc}")

    # ------------------------------------------------------------------
    # 5. Failure Analysis
    # ------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # We need to gather features, predictions, and targets to correlate errors
    # Load feature names from metadata
    meta_path = os.path.join(Config.WORKING_DIR, "feature_meta.joblib")
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        feature_cols = meta["feature_cols"]
    else:
        print(
            "Warning: Feature metadata not found. Skipping detailed feature name reporting."
        )
        feature_cols = [f"feat_{i}" for i in range(input_dim)]

    all_preds_prob = []
    all_targets = []
    all_features = []

    # Collect data (disable gradients for speed)
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            all_preds_prob.append(probs)
            all_targets.append(labels.view(-1).cpu().numpy())
            # Move features to CPU for correlation calculation
            all_features.append(features.cpu().numpy())

    all_preds_prob = np.concatenate(all_preds_prob)
    all_targets = np.concatenate(all_targets)
    all_features = np.concatenate(all_features, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds_prob)

    # Calculate correlation between Error and each Feature
    # We use pandas for convenient correlation computation
    # Construct a DataFrame for analysis (subset if too large to save memory, but 800k rows is fine)

    # To save memory/time, we can compute correlation manually or use a subset
    # Let's use a subset of 100k samples if validation is huge
    if len(errors) > 100000:
        indices = np.random.choice(len(errors), 100000, replace=False)
        errors_sub = errors[indices]
        features_sub = all_features[indices]
    else:
        errors_sub = errors
        features_sub = all_features

    print(f"Calculating correlations on {len(errors_sub)} samples...")

    correlations = []
    for i in range(input_dim):
        # Pearson correlation between feature column i and error
        feat_col = features_sub[:, i]
        # Handle constant features (std=0) to avoid NaN
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors_sub)[0, 1]

        correlations.append((feature_cols[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error (Correlation):")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # ------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------
    TARGET_SCORE = 0.62458462731896

    if final_val_mcc > TARGET_SCORE:
        print(
            f"\nMetric ({final_val_mcc}) > Threshold ({TARGET_SCORE}). Generating Submission..."
        )
        generate_submission(model, scaler, best_threshold, device)
    else:
        print(
            f"\nMetric ({final_val_mcc}) <= Threshold ({TARGET_SCORE}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
