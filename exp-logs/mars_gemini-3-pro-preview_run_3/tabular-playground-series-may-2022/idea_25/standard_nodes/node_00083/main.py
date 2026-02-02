import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import process_data, ManufacturingDataset
from library.model import MORPE
from library.train_eval import train_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Use Config epochs to avoid premature truncation (Cite Lesson 00080)
    FAST_EPOCHS = Config.EPOCHS

    # 2. Data Processing
    print("Initializing data pipeline...")
    # Load cached data if available, otherwise process from scratch
    df_train, df_val, df_test, meta_dict = process_data(load_cached_data=True)

    cat_cols = meta_dict["cat_cols"]
    cont_cols = meta_dict["cont_cols"]
    vocab_sizes_dict = meta_dict["vocab_sizes"]
    vocab_sizes = [vocab_sizes_dict[c] for c in cat_cols]

    # 3. Datasets and Loaders
    print("Creating dataloaders...")
    train_ds = ManufacturingDataset(df_train, cat_cols, cont_cols, "target")
    val_ds = ManufacturingDataset(df_val, cat_cols, cont_cols, "target")
    test_ds = ManufacturingDataset(df_test, cat_cols, cont_cols, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing MORPE model...")
    model = MORPE(
        vocab_sizes_list=vocab_sizes,
        num_cont=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.STREAMS,
    ).to(device)

    # 5. Optimizer Configuration
    # Use Uniform Weight Decay (1e-5) with AdamW (Cite Lesson 00082)
    print("Configuring optimizer with uniform weight decay...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.MAX_LR, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=FAST_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print(f"Starting training for {FAST_EPOCHS} epochs on {device}...")
    best_auc = 0.0
    best_model_state = None

    for epoch in range(FAST_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.6f} | Val AUC: {val_auc:.10f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()

    # 7. Final Evaluation & Metrics
    print(f"Final Validation Metric: {best_auc}")

    # Restore best model for analysis and prediction
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    model.eval()

    # Generate predictions on validation set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for cat, cont, target in val_loader:
            cat, cont = cat.to(device), cont.to(device)
            outputs = model(cat, cont)
            # Average probabilities across streams
            probs = torch.stack([torch.sigmoid(out) for out in outputs]).mean(dim=0)
            val_probs.append(probs.cpu().numpy())
            val_targets.append(target.numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Prepare dataframe for correlation analysis
    # Use the processed validation dataframe which contains features
    analysis_df = df_val.copy()

    # Remove non-feature columns
    cols_to_drop = ["id", "target", "source_path"]
    analysis_df = analysis_df.drop(
        columns=[c for c in cols_to_drop if c in analysis_df.columns]
    )

    # Add error column
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    # We use standard correlation. For ordinal encoded categoricals, this gives a rough proxy of linear relationship.
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.head(5))

    # 9. Submission
    THRESHOLD = 0.9975746465492954

    if best_auc > THRESHOLD:
        print(f"\nValidation metric {best_auc} > {THRESHOLD}. Generating submission...")

        test_preds = predict(model, test_loader, device)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission = pd.DataFrame({"id": df_test["id"], "target": test_preds})
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric {best_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
