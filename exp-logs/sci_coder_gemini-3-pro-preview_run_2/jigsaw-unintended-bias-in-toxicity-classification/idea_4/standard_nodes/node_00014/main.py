import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.config import (
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    MODEL_NAME,
    DROPOUT,
    IDENTITY_COLUMNS,
    AUX_TOXICITY_COLUMNS,
    LR,
    NUM_EPOCHS,
    SAVED_MODEL_PATH,
    VAL_METADATA_PATH,
)
from library.utils import seed_everything, JigsawEvaluator
from library.data import ToxicityDataset
from library.model import MultiTaskRoBERTa
from library.engine import (
    train_fn,
    eval_fn,
    inference_fn,
    save_submission,
    EarlyStopping,
)


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Setup Environment
    seed_everything(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # Limit training data size to ensure fast baseline execution as per requirements
    TRAIN_DEBUG_SIZE = 100000
    print(f"Loading datasets (Train limit: {TRAIN_DEBUG_SIZE})...")

    # Train set with limit
    train_ds = ToxicityDataset(
        "train", load_cached_data=True, debug_size=TRAIN_DEBUG_SIZE
    )
    # Full validation set is required for accurate metric calculation
    val_ds = ToxicityDataset("validation", load_cached_data=True)
    # Full test set for submission
    test_ds = ToxicityDataset("test", load_cached_data=True)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,  # Larger batch size for inference
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    # 3. Model Initialization
    print("Initializing Multi-Task RoBERTa model...")
    model = MultiTaskRoBERTa(
        model_name=MODEL_NAME,
        dropout_rate=DROPOUT,
        num_identities=len(IDENTITY_COLUMNS),
    )
    model.to(DEVICE)

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # OneCycleLR for fast convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # 5. Training Loop
    early_stopping = EarlyStopping(patience=2, mode="max", save_path=SAVED_MODEL_PATH)

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train Step
        train_loss = train_fn(train_loader, model, optimizer, DEVICE, scheduler)

        # Validation Step
        val_preds, val_targets, val_identities = eval_fn(val_loader, model, DEVICE)

        # Calculate Metric
        val_identities_df = pd.DataFrame(val_identities, columns=IDENTITY_COLUMNS)
        evaluator = JigsawEvaluator(val_targets, val_preds, val_identities_df)
        final_score, overall_auc, sub_auc, bpsn_auc, bnsp_auc = (
            evaluator.get_final_metric()
        )

        print(f"Validation Score: {final_score:.6f}")
        print(f"  Overall AUC:  {overall_auc:.6f}")
        print(f"  Subgroup AUC: {sub_auc:.6f}")
        print(f"  BPSN AUC:     {bpsn_auc:.6f}")
        print(f"  BNSP AUC:     {bnsp_auc:.6f}")

        # Check Early Stopping
        early_stopping(final_score, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 6. Final Evaluation & Failure Analysis
    print("\n=== Final Evaluation & Failure Analysis ===")

    # Load the best model saved by EarlyStopping
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    model.to(DEVICE)

    # Re-evaluate on full validation set
    val_preds, val_targets, val_identities = eval_fn(val_loader, model, DEVICE)
    val_identities_df = pd.DataFrame(val_identities, columns=IDENTITY_COLUMNS)
    evaluator = JigsawEvaluator(val_targets, val_preds, val_identities_df)
    final_score, _, _, _, _ = evaluator.get_final_metric()

    # Print the required metric format
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis: Correlation of Error with Features
    targets_np = np.array(val_targets)
    preds_np = np.array(val_preds)
    errors = np.abs(targets_np - preds_np)

    # Load validation metadata to get auxiliary columns for analysis
    # The order of val_loader (sequential) matches the metadata file
    val_meta_df = pd.read_csv(VAL_METADATA_PATH)

    if len(val_meta_df) == len(errors):
        val_meta_df["error"] = errors

        # Select columns to analyze (Identities + Toxicity Subtypes)
        analysis_cols = IDENTITY_COLUMNS + AUX_TOXICITY_COLUMNS
        # Filter for columns present in the dataframe
        analysis_cols = [c for c in analysis_cols if c in val_meta_df.columns]

        print("\nCorrelation between Error Magnitude and Features:")
        correlations = (
            val_meta_df[analysis_cols]
            .corrwith(val_meta_df["error"])
            .sort_values(ascending=False)
        )
        print(correlations)
    else:
        print(
            f"Warning: Metadata length ({len(val_meta_df)}) does not match predictions ({len(errors)}). Skipping correlation analysis."
        )

    # 7. Submission Generation
    THRESHOLD = 0.9273793163893314

    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        test_ids, test_preds = inference_fn(test_loader, model, DEVICE)
        save_submission(test_ids, test_preds)
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
