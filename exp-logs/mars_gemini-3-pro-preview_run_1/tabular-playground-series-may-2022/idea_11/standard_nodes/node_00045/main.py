import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import provided library modules
from library.config import Config, set_seed
from library.data import process_data, ManufacturingDataset
from library.model import ResDeGUT
from library.engine import train_fn, eval_fn, predict_fn


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Using device: {Config.DEVICE}")

    # 2. Data Loading
    # process_data handles caching and loading from metadata
    data, vocab_size = process_data(load_cached_data=True)

    # 3. Datasets & Loaders
    # Train dataset with masking
    train_dataset = ManufacturingDataset(
        data["X_num_train"],
        data["X_seq_train"],
        data["y_train"],
        mask_prob=Config.MASK_PROB,
    )

    # Validation dataset (no masking)
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"], mask_prob=0.0
    )

    # Test dataset (no masking, no targets)
    test_dataset = ManufacturingDataset(
        data["X_num_test"], data["X_seq_test"], None, mask_prob=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    num_features = data["X_num_train"].shape[1]
    seq_len = data["X_seq_train"].shape[1]

    model = ResDeGUT(num_features, seq_len, vocab_size, Config).to(Config.DEVICE)

    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-2)

    # OneCycleLR requires total steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    # 6. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, Config.DEVICE, Config
        )

        # Validate
        val_loss, val_auc = eval_fn(model, val_loader, Config.DEVICE)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.5f}"
        )

        # Save Best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print("  -> New Best Model Saved")

    # 7. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    # Compute final metric on full validation set
    _, final_val_auc = eval_fn(model, val_loader, Config.DEVICE)
    print(f"Final Validation Metric: {final_val_auc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get raw predictions for validation set
    val_preds = predict_fn(model, val_loader, Config.DEVICE)
    y_val = data["y_val"]

    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # Correlate errors with numerical features
    X_val = data["X_num_val"]
    correlations = []

    # We don't have feature names easily accessible in the numpy arrays,
    # but we can use indices.
    for i in range(X_val.shape[1]):
        # Calculate Pearson correlation
        if np.std(X_val[:, i]) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(X_val[:, i], errors)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: Correlation = {corr:.4f}")

    # 9. Submission
    THRESHOLD = 0.9977872734278943

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = predict_fn(model, test_loader, Config.DEVICE)

        submission = pd.DataFrame({"id": data["ids_test"], "target": test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
