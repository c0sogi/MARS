import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_processing import preprocess_features, ManufacturingDataset
from library.model import ParallelFunnelEnsemble, set_seed
from library.training import train_epoch, validate, inference


def perform_failure_analysis(model, val_loader, val_df, cont_cols, cat_cols, device):
    """
    Analyzes model errors on the validation set.
    Computes correlation between absolute error and input features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    # Get predictions and targets
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            mean_probs = torch.mean(probs, dim=1)

            val_preds.append(mean_probs.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Calculate Error
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for correlation analysis
    # We use the processed val_df which lines up with the loader if shuffle=False
    analysis_df = val_df[cont_cols + cat_cols].copy()
    analysis_df["error"] = errors

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")

    # Sort by absolute correlation
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error:")
    print(sorted_corrs.head(5))

    return val_targets, val_preds


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading and processing data...")
    train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=True, config=Config
    )

    # Create Datasets
    train_dataset = ManufacturingDataset(train_df, cat_cols, cont_cols, mode="train")
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, mode="val")
    # Test dataset will be created only if needed

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SSPFE Model...")
    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Final Evaluation & Failure Analysis
    print("\nTraining complete.")

    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Re-validate to ensure we have the metric for the best weights
    final_val_auc = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc:.16f}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, val_df, cont_cols, cat_cols, device)

    # 7. Submission Logic
    THRESHOLD = 0.9975746465492954

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        final_preds = inference(model, test_loader, device)

        submission = pd.DataFrame({"id": test_df["id"], "target": final_preds})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation metric ({final_val_auc:.6f}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
