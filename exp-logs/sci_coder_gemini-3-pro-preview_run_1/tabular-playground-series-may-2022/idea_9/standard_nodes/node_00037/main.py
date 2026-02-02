import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.architecture import DiGUT
from library.data_factory import preprocess_data, ManufacturingDataset
from library.trainer import train_one_epoch, evaluate


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Configuration and Setup
    config = Config()

    # Fast Baseline Overrides:
    # Reduce epochs to ensure execution within time limits while maintaining convergence
    config.EPOCHS = 15
    # Scale Batch Size to Match Available GPU Memory (Cite debug_lesson_3)
    config.BATCH_SIZE = 1024

    # Ensure output directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    set_seed(config.SEED)
    print(f"Running on device: {config.DEVICE}")
    config.display()

    # 2. Data Loading
    print("\nLoading and preprocessing data...")
    (
        X_num_train,
        X_seq_train,
        y_train,
        X_num_val,
        X_seq_val,
        y_val,
        X_num_test,
        X_seq_test,
        ids_test,
        meta,
    ) = preprocess_data(config, load_cached_data=True)

    # 3. Dataset and DataLoader Creation
    # Note: is_train=False for train_dataset because noise is applied in the training loop on GPU
    train_dataset = ManufacturingDataset(
        X_num_train, X_seq_train, y_train, is_train=False, config=config
    )
    val_dataset = ManufacturingDataset(
        X_num_val, X_seq_val, y_val, is_train=False, config=config
    )
    test_dataset = ManufacturingDataset(
        X_num_test, X_seq_test, None, is_train=False, config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("\nInitializing DiGUT model...")
    model = DiGUT(
        num_numerical_features=meta["num_numerical_features"],
        vocab_size=meta["vocab_size"],
        sequence_length=meta["sequence_length"],
        config=config,
    ).to(config.DEVICE)

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.PCT_START,
        anneal_strategy="cos",
    )

    # 6. Training Loop
    print("\nStarting training...")
    best_auc = 0.0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE, config
        )

        # Validate
        val_auc, _ = evaluate(model, val_loader, config.DEVICE)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.5f} | Val AUC: {val_auc:.6f} | Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    # 7. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )

    final_val_auc, val_preds = evaluate(model, val_loader, config.DEVICE)
    # Ensure 1D array for preds
    val_preds = val_preds.flatten()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error magnitude
    errors = np.abs(y_val - val_preds)

    # Create DataFrame for correlation analysis
    # X_num_val is standardized, but correlation is scale-invariant
    feature_names = meta["num_cols"]
    df_analysis = pd.DataFrame(X_num_val, columns=feature_names)
    df_analysis["error_magnitude"] = errors

    # Compute correlation
    corr_series = df_analysis.corr()["error_magnitude"].drop("error_magnitude")
    corr_abs = corr_series.abs().sort_values(ascending=False)

    print("Top 5 features correlated with error magnitude:")
    print(corr_series.loc[corr_abs.index[:5]])

    # 9. Submission Generation
    THRESHOLD = 0.9977872734278943

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Inference on Test Set
        _, test_preds = evaluate(model, test_loader, config.DEVICE)
        test_preds = test_preds.flatten()

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": ids_test, "target": test_preds})

        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
