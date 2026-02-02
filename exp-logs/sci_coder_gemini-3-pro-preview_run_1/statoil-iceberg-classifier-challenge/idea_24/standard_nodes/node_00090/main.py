"""
Main execution script for Iceberg Classification Task.
Implements Idea 24: Calibrated SAM-SWA ResNet Ensemble with Dual-Domain Augmentation.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import library modules
from library import config, utils, data_factory, model_factory, engine, sam, workflow


def main():
    # 1. Setup
    utils.seed_everything()
    device = utils.get_device()
    print(f"Using device: {device}")

    # 2. Phase 1: Calibration
    # Determines the optimal convergence epoch using 5-fold CV
    print("\n=== Phase 1: Calibration ===")
    # This runs the CV process defined in workflow.py and returns the average best epoch
    optimal_epochs = workflow.run_phase_1_calibration(n_splits=5)
    print(f"Optimal Convergence Epoch (E_conv): {optimal_epochs}")

    # 3. Validation Run
    # Train a single model on the training split and evaluate on the hold-out validation split
    # This ensures we have a valid metric and can perform failure analysis before full production training.
    print("\n=== Phase 1.5: Validation Run ===")

    # Load Metadata to get splits
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(config.VAL_META_PATH)

    train_idxs = df_train_meta["sample_index"].values
    val_idxs = df_val_meta["sample_index"].values

    # Get DataLoaders
    # We load cached data to speed things up
    train_loader, val_loader, _ = data_factory.get_dataloaders(
        train_idxs=train_idxs, val_idxs=val_idxs, load_cached_data=True
    )

    # Initialize Model
    model = model_factory.get_model()

    # Optimizer (SAM)
    base_optimizer = torch.optim.AdamW
    optimizer = sam.SAM(
        model.parameters(),
        base_optimizer,
        rho=config.SAM_RHO,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.base_optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )

    # Loss (Smoothed for training)
    train_criterion = workflow.SmoothBCEWithLogitsLoss(smoothing=config.LABEL_SMOOTHING)

    # Engine
    trainer = engine.Engine(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=train_criterion,
        scheduler=scheduler,
    )

    # Train Loop (SAM Phase)
    print(f"Training validation model for {optimal_epochs} epochs...")
    for epoch in range(1, optimal_epochs + 1):
        loss, acc = trainer.train_one_epoch(train_loader, epoch)
        # Step scheduler on training loss to mimic production dynamics
        scheduler.step(loss)

    # SWA Transition Phase
    print(f"Transitioning to SWA for {config.SWA_DURATION_EPOCHS} epochs...")
    swa_handler = engine.SWAHandler(model, device)

    # Set SWA LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.SWA_LR

    for i in range(config.SWA_DURATION_EPOCHS):
        current_epoch = optimal_epochs + i + 1
        loss, acc = trainer.train_one_epoch(train_loader, current_epoch)
        swa_handler.update(model)

    # Update BN Statistics
    print("Updating SWA BN statistics...")
    swa_handler.update_bn(train_loader)

    # Get Final Validation Model
    val_model = swa_handler.get_model()

    # Evaluate
    print("Evaluating on Validation Set...")
    # Swap model in trainer to use the SWA model for TTA
    trainer.model = val_model

    # validate_tta returns (avg_loss, avg_acc, preds, targets)
    # preds are probabilities
    _, _, val_preds, val_targets = trainer.validate_tta(val_loader)

    # Compute Final Metric (Log Loss)
    final_metric = log_loss(val_targets, val_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Get Incidence Angles for Validation Set
    # Note: The loader returns normalized angles.
    # We want to check correlation. Normalization doesn't affect correlation.
    val_angles_list = []
    # Iterate loader deterministically (shuffle=False for val_loader)
    for _, angles, _ in val_loader:
        val_angles_list.extend(angles.numpy())
    val_angles = np.array(val_angles_list)

    # Compute Correlation
    if len(val_angles) == len(errors):
        corr = np.corrcoef(errors, val_angles)[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr:.6f}")
    else:
        print(
            "Warning: Mismatch in lengths for failure analysis. Skipping correlation."
        )

    # 5. Submission
    THRESHOLD = 0.16918645240183008

    if final_metric < THRESHOLD:
        print(f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}).")
        print("=== Phase 2: Production Training ===")
        # Train ensemble on FULL data (Train + Val)
        workflow.run_phase_2_production(optimal_epochs, n_models=5)

        print("\n=== Generating Submission ===")
        workflow.generate_submission(n_models=5)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
