import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library import data, model, loss, engine, utils


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    utils.seed_everything(Config.SEED)
    device = utils.get_device()
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading DataLoaders...")
    # Load cached data to save time
    train_loader_full, val_loader, test_loader, test_ids = data.get_dataloaders(
        load_cached_data=True
    )

    # Subsample training data to ensure execution within time limits (Fast Baseline)
    # We use 150,000 samples which fits comfortably within 2 hours on A100
    N_TRAIN_SAMPLES = 150000
    train_dataset = train_loader_full.dataset
    total_samples = len(train_dataset)

    if total_samples > N_TRAIN_SAMPLES:
        print(
            f"Subsampling training set from {total_samples} to {N_TRAIN_SAMPLES} samples."
        )
        # Create random indices
        indices = torch.randperm(total_samples)[:N_TRAIN_SAMPLES]
        train_subset = Subset(train_dataset, indices)

        # Re-create DataLoader for the subset
        train_loader = DataLoader(
            train_subset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
        )
    else:
        train_loader = train_loader_full

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing MultiTaskRoBERTa model...")
    main_model = model.MultiTaskRoBERTa(Config)
    main_model.to(device)

    # Initialize SWA Model
    if Config.USE_SWA:
        print("Initializing SWA model...")
        swa_model = AveragedModel(main_model)
    else:
        swa_model = None

    # --------------------------------------------------------------------------
    # 4. Optimizer, Scheduler & Loss
    # --------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        main_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Total steps calculation
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if Config.USE_SWA:
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    loss_fn = loss.HybridContrastiveLoss()

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    best_score = -float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        avg_train_loss = engine.train_one_epoch(
            main_model, train_loader, optimizer, scheduler, device, loss_fn
        )
        print(f"  Average Train Loss: {avg_train_loss:.4f}")

        # SWA Update
        if Config.USE_SWA and (epoch + 1) >= Config.SWA_START_EPOCH:
            print("  Updating SWA parameters...")
            swa_model.update_parameters(main_model)
            swa_scheduler.step()

        # Validation
        avg_val_loss, metrics = engine.valid_fn(main_model, val_loader, device, loss_fn)
        current_score = metrics["score"]

        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Score: {current_score:.4f}")
        print(f"  Overall AUC: {metrics['overall_auc']:.4f}")

        # Save Best Model
        if current_score > best_score:
            print(
                f"  New best score! ({best_score:.4f} -> {current_score:.4f}). Saving model."
            )
            best_score = current_score
            torch.save(main_model.state_dict(), Config.MODEL_SAVE_PATH)

    # --------------------------------------------------------------------------
    # 6. Final Evaluation
    # --------------------------------------------------------------------------
    print("\nTraining Complete. Preparing for Final Evaluation...")

    # Select Final Model (SWA or Best)
    final_model = main_model

    if Config.USE_SWA and swa_model is not None:
        print("Using SWA model for final inference.")
        # We use the AveragedModel wrapper directly
        final_model = swa_model
        # Ensure it's on the correct device
        final_model.to(device)
        # Save SWA weights
        torch.save(final_model.state_dict(), Config.SWA_MODEL_SAVE_PATH)
    else:
        print("Loading best model checkpoint for final inference.")
        final_model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
        final_model.to(device)

    # Final Validation on full set
    _, final_metrics = engine.valid_fn(final_model, val_loader, device, loss_fn)
    final_score_val = final_metrics["score"]

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_score_val:.16f}")

    # --------------------------------------------------------------------------
    # 7. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Generate raw predictions for validation set
    val_preds = engine.inference_fn(final_model, val_loader, device)

    # Retrieve targets and identities from validation loader
    val_targets = []
    val_identities = []

    # Iterate loader to get ground truth (order is preserved if shuffle=False)
    for batch in val_loader:
        val_targets.append(batch["target"].numpy())
        if "identities" in batch:
            val_identities.append(batch["identities"].numpy())

    val_targets = np.concatenate(val_targets)
    val_identities = np.concatenate(val_identities, axis=0)

    # Calculate Absolute Error
    errors = np.abs(val_targets - val_preds)

    # Build Analysis DataFrame
    analysis_df = pd.DataFrame({"error": errors})
    for i, col_name in enumerate(Config.IDENTITY_COLUMNS):
        # Handle case where batch might not cover all cols (unlikely with fixed schema)
        if i < val_identities.shape[1]:
            analysis_df[col_name] = val_identities[:, i]

    # Compute Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    # Filter out the error-error correlation
    correlations = correlations.drop("error", errors="ignore")

    print("Correlation between Error Magnitude and Identity Attributes:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 8. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9273793163893314

    if final_score_val > THRESHOLD:
        print(
            f"\nFinal Score ({final_score_val:.6f}) > Threshold ({THRESHOLD:.6f}). Generating Submission..."
        )

        test_preds = engine.inference_fn(final_model, test_loader, device)

        submission = pd.DataFrame({"id": test_ids, "prediction": test_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nFinal Score ({final_score_val:.6f}) <= Threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
