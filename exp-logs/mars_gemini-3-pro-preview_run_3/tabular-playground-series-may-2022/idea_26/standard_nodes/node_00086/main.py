import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.dataset import ManufacturingDataset
from library.model import RPFEModel
from library.engine import Engine


def main():
    # 1. Setup & Configuration
    # Override Config for Fast Baseline
    EPOCHS = 10
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    THRESHOLD = 0.9975746465492954

    # Initialize Engine (handles seeding and device)
    engine = Engine()
    engine.set_seed(Config.SEED)
    print(f"Device: {engine.device}")

    # 2. Data Loading
    print("Loading and processing data...")
    fe = FeatureEngineer()
    # Use cached data if available for speed
    train_df, val_df, test_df, vocab_sizes = fe.process_data(load_cached_data=True)

    # Identify columns
    cat_cols = [f"f_27_{i}" for i in range(10)]
    exclude_cols = {"id", "target", "source_path"} | set(cat_cols)
    cont_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Continuous features: {len(cont_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    # Create Datasets
    train_ds = ManufacturingDataset(train_df, cat_cols, cont_cols, is_test=False)
    val_ds = ManufacturingDataset(val_df, cat_cols, cont_cols, is_test=False)
    test_ds = ManufacturingDataset(test_df, cat_cols, cont_cols, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RPFEModel(vocab_sizes, len(cont_cols))
    model.to(engine.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = engine.train_fn(
            model, train_loader, optimizer, scheduler, criterion
        )
        val_auc = engine.eval_fn(model, val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("Training complete.")

    # 5. Validation Assessment
    print("Loading best model for validation assessment...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=engine.device)
    )

    # Calculate Final Metric
    final_val_auc = engine.eval_fn(model, val_loader)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()
    val_preds = []
    val_targets = []

    # Get raw predictions and targets
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(engine.device)
            categorical = batch["categorical"].to(engine.device)
            targets = batch["target"].to(engine.device)

            outputs = model(continuous, categorical)
            probs = torch.sigmoid(outputs).mean(dim=1)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets).ravel()

    # Calculate Error
    errors = np.abs(val_targets - val_preds)

    # Create analysis dataframe
    # We use the original val_df for features, but we need to ensure alignment.
    # val_loader is sequential (shuffle=False), so order is preserved.
    analysis_df = val_df.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlations
    # Select numeric columns for correlation (continuous + encoded categorical)
    numeric_cols = cont_cols + cat_cols
    correlations = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop 5 Features correlated with Error Magnitude:")
    print(correlations.head(5))

    # 7. Submission
    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )

        # Make predictions on test set
        test_preds = engine.predict_fn(model, test_loader)

        # Load sample submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Ensure directory exists
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # Assign predictions
        sample_sub["target"] = test_preds

        # Save
        sample_sub.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric {final_val_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
