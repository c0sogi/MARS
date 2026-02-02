import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

import library.config as config
from library.data_factory import get_dataloaders
from library.model_factory import get_model
from library.engine import set_seed, train_epoch, evaluate, predict_with_tta


def perform_failure_analysis(val_df, val_targets, val_probs):
    """
    Analyzes the correlation between prediction error and input data characteristics.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Absolute Error
    errors = np.abs(val_targets - val_probs)
    val_df["error"] = errors

    # Extract structural features (Slice Counts)
    # We iterate through the dataframe to count files in the directories
    # This helps determine if scan depth affects model performance
    feature_correlations = {}
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    print("Extracting metadata features for analysis...")
    for mod in modalities:
        col_name = f"path_{mod}"
        counts = []
        for _, row in val_df.iterrows():
            full_path = os.path.join(config.INPUT_DIR, row[col_name])
            try:
                # Fast directory listing
                num_files = len(
                    [name for name in os.listdir(full_path) if name.endswith(".dcm")]
                )
            except Exception:
                num_files = 0
            counts.append(num_files)

        val_df[f"{mod}_count"] = counts

        # Compute correlation
        if np.std(counts) > 0:  # Avoid division by zero
            corr, _ = pearsonr(errors, counts)
            feature_correlations[f"{mod}_count"] = corr
        else:
            feature_correlations[f"{mod}_count"] = 0.0

    print("Correlation between Error Magnitude and Input Features:")
    for feature, corr in feature_correlations.items():
        print(f"  {feature}: {corr:.4f}")

    # Identify worst failures
    print("\nTop 5 Worst Predictions:")
    worst_indices = np.argsort(errors)[-5:][::-1]
    for idx in worst_indices:
        row = val_df.iloc[idx]
        print(
            f"  ID: {row['BraTS21ID']}, True: {val_targets[idx]:.1f}, Pred: {val_probs[idx]:.4f}, Error: {errors[idx]:.4f}"
        )


def main():
    # 1. Setup
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Data
    # Using load_cached_data=True to leverage pre-processing
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model & Optimizer
    print("Initializing model...")
    model = get_model()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    # We use the config's epoch count (15) which is appropriate for a fast baseline
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. Final Validation Metric
    # Required output format
    print(f"Final Validation Metric: {best_auc}")

    # 6. Failure Analysis
    # Load best model for analysis
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    model.eval()

    # Collect validation predictions
    val_targets = []
    val_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_targets.extend(targets.numpy())
            val_probs.extend(probs)

    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)

    # Load validation metadata to link features to errors
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Ensure data alignment (Loader is sequential, DF is sequential)
    if len(val_df) == len(val_targets):
        perform_failure_analysis(val_df, val_targets, val_probs)
    else:
        print(
            "Warning: Validation dataframe length does not match prediction count. Skipping detailed failure analysis."
        )

    # 7. Conditional Submission
    submission_threshold = 0.6321818181818182

    if best_auc > submission_threshold:
        print(
            f"\nValidation metric ({best_auc:.6f}) exceeds threshold ({submission_threshold}). Generating submission..."
        )
        predict_with_tta(model, test_loader, device, config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({best_auc:.6f}) did not exceed threshold ({submission_threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
