import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.dataset import get_dataloaders
from library.model import DCNv2
from library.engine import Engine


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    # 15 epochs is sufficient for convergence with OneCycleLR on this dataset size
    Config.EPOCHS = 15

    # Set seeds
    set_seed(Config.SEED)

    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Device selected: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load data using the library function
    # load_cached_data=True to use ./working parquet files if available
    # debug=False to use the full dataset (A100 is fast enough)
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = DCNv2().to(device)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    engine = Engine(model, device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    best_auc = 0.0
    patience_counter = 0

    # Ensure model save path exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = engine.train_one_epoch(
            train_loader, optimizer, scheduler, criterion
        )

        # Validate
        val_loss, val_auc = engine.evaluate(val_loader, criterion)

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Val AUC = {val_auc:.6f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # ==========================================
    # 5. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis on Validation Set...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect predictions, targets, and features for analysis
    all_targets = []
    all_preds = []
    all_cont_feats = []

    # We iterate manually to capture features which are not returned by engine.evaluate
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            outputs = model(continuous, categorical)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_cont_feats.append(continuous.cpu().numpy())

    # Concatenate results
    all_targets = np.concatenate(all_targets).ravel()
    all_preds = np.concatenate(all_preds).ravel()
    all_cont_feats = np.vstack(all_cont_feats)

    # Calculate Final Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Features
    errors = np.abs(all_targets - all_preds)

    # Define feature names
    # Continuous: Base cols + unique_character_count
    cont_feature_names = Config.BASE_CONT_COLS + ["unique_character_count"]

    print("\nCorrelation between Absolute Error and Continuous Features:")
    correlations = []
    for i, name in enumerate(cont_feature_names):
        if i < all_cont_feats.shape[1]:
            feat_values = all_cont_feats[:, i]
            # Handle potential constant values to avoid warnings
            if np.std(feat_values) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(errors, feat_values)
            correlations.append((name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"{name}: {corr:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.9971550270448856

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Use engine's predict method for test set
        ids, preds = engine.predict(test_loader)

        submission_df = pd.DataFrame(
            {Config.ID_COL: ids.astype(int), Config.TARGET_COL: preds}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
