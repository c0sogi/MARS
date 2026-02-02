import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_logger, save_checkpoint
from library.data import CassavaDataset, get_transforms, create_loaders
from library.network import get_model, ModelEMA
from library.trainer import train_one_epoch, validate
from library.inference import predict_fold, inference


def run_training(logger):
    """
    Executes the 5-Fold training loop with Progressive Resizing.
    """
    logger.info(f"Starting Training with Idea 14 configuration.")
    logger.info(f"Debug Mode: {CFG.debug}")
    logger.info(
        f"Phase 1: {CFG.p1_epochs} epochs @ {CFG.p1_img_size}x{CFG.p1_img_size}"
    )
    logger.info(
        f"Phase 2: {CFG.p2_epochs} epochs @ {CFG.p2_img_size}x{CFG.p2_img_size}"
    )

    # 1. Load Training Metadata
    train_df = pd.read_csv(CFG.train_csv)

    # Apply debug sampling if configured
    if CFG.debug:
        train_df = train_df.sample(
            n=min(len(train_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)
        logger.info(f"Debug mode: Subsampled training data to {len(train_df)} rows.")

    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # ==========================
        # Phase 1: Coarse Training
        # ==========================
        logger.info(f"--- Phase 1: {CFG.p1_img_size}x{CFG.p1_img_size} ---")

        train_loader, val_loader = create_loaders(
            fold_train_df, fold_val_df, CFG.p1_img_size, CFG.p1_batch_size
        )

        model = get_model(pretrained=True)
        model_ema = ModelEMA(model) if CFG.use_ema else None

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.p1_epochs, eta_min=CFG.min_lr
        )

        best_acc = 0.0
        phase1_weights_path = os.path.join(CFG.working_dir, f"phase1_fold_{fold}.pth")

        for epoch in range(CFG.p1_epochs):
            train_loss, train_acc = train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                CFG.device,
                scheduler,
                model_ema,
                logger,
            )
            val_loss, val_acc = validate(
                model_ema.module if model_ema else model, val_loader, CFG.device, logger
            )

            if scheduler:
                scheduler.step()

            # Save best Phase 1 model (for Phase 2 initialization)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    (model_ema.module if model_ema else model).state_dict(),
                    phase1_weights_path,
                )

        # Clean up to save memory
        del model, model_ema, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

        # ==========================
        # Phase 2: Fine-Tuning
        # ==========================
        logger.info(f"--- Phase 2: {CFG.p2_img_size}x{CFG.p2_img_size} ---")

        # Re-create loaders with larger image size
        train_loader, val_loader = create_loaders(
            fold_train_df, fold_val_df, CFG.p2_img_size, CFG.p2_batch_size
        )

        # Initialize model and load Phase 1 weights
        model = get_model(pretrained=False)
        if os.path.exists(phase1_weights_path):
            state_dict = torch.load(phase1_weights_path, map_location=CFG.device)
            model.load_state_dict(state_dict)
            logger.info("Loaded Phase 1 weights.")
        else:
            logger.warning("Phase 1 weights not found. Starting Phase 2 from scratch.")

        model_ema = ModelEMA(model) if CFG.use_ema else None

        # Re-initialize optimizer for fine-tuning
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.p2_epochs, eta_min=CFG.min_lr
        )

        # Reset best metric for this phase
        best_acc = 0.0
        final_checkpoint_path = os.path.join(
            CFG.output_dir, f"checkpoint_fold_{fold}.pth"
        )

        for epoch in range(CFG.p2_epochs):
            train_loss, train_acc = train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                CFG.device,
                scheduler,
                model_ema,
                logger,
            )
            val_loss, val_acc = validate(
                model_ema.module if model_ema else model, val_loader, CFG.device, logger
            )

            if scheduler:
                scheduler.step()

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": (
                        model_ema.module if model_ema else model
                    ).state_dict(),
                    "best_acc": best_acc,
                },
                is_best=(val_acc > best_acc),
                filepath=final_checkpoint_path,
            )

            if val_acc > best_acc:
                best_acc = val_acc

        # Cleanup
        del model, model_ema, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()


def run_holdout_validation(logger):
    """
    Performs ensemble inference on the hold-out validation set.
    """
    logger.info("\nStarting Hold-out Validation...")

    # Load Hold-out Validation Data
    val_df = pd.read_csv(CFG.val_csv)
    if CFG.debug:
        val_df = val_df.sample(
            n=min(len(val_df), CFG.debug_sample_size), random_state=CFG.seed
        ).reset_index(drop=True)

    # Create DataLoader (using Phase 2 settings)
    val_ds = CassavaDataset(val_df, transform=get_transforms("val", CFG.p2_img_size))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CFG.p2_batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    fold_probs = []

    # Ensemble predictions from all folds
    for fold in range(CFG.n_folds):
        # We reuse predict_fold from inference library
        # It expects a test_loader (returns images, _), which matches val_loader structure
        probs = predict_fold(fold, val_loader, CFG.device)
        if probs is not None:
            fold_probs.append(probs)

    if not fold_probs:
        logger.error("No predictions generated.")
        return 0.0, None, None, None

    # Average probabilities
    avg_probs = torch.stack(fold_probs).mean(dim=0)
    preds = torch.argmax(avg_probs, dim=1).numpy()
    targets = val_df["label"].values

    # Calculate Metric
    acc = accuracy_score(targets, preds)
    logger.info(f"Final Validation Metric: {acc}")
    print(f"Final Validation Metric: {acc}")  # Explicit print as requested

    return acc, avg_probs.numpy(), targets, val_df


def run_failure_analysis(probs, targets, df, logger):
    """
    Analyzes correlation between error magnitude and file size.
    """
    logger.info("\nRunning Failure Analysis...")

    # Calculate Error Magnitude
    # Error = 1.0 - Probability assigned to the correct class
    n_samples = len(targets)
    error_magnitudes = []
    file_sizes = []

    for i in range(n_samples):
        true_label = targets[i]
        prob_true = probs[i, true_label]
        error = 1.0 - prob_true
        error_magnitudes.append(error)

        # Get file size
        rel_path = df.iloc[i]["file_path"]
        full_path = os.path.join(CFG.input_dir, rel_path)
        try:
            size = os.path.getsize(full_path)
            file_sizes.append(size)
        except:
            file_sizes.append(0)

    # Calculate Correlation
    if len(error_magnitudes) > 1:
        corr, _ = pearsonr(error_magnitudes, file_sizes)
        logger.info(f"Correlation between Error Magnitude and File Size: {corr:.4f}")
        print(f"Correlation between Error Magnitude and File Size: {corr:.4f}")
    else:
        logger.warning("Not enough samples for correlation analysis.")


def main():
    # Setup
    seed_everything(CFG.seed)
    logger = get_logger()

    # Override Configuration for Fast Baseline Execution
    # We enable debug mode to limit samples and reduce epochs drastically
    # to ensure the script completes within the time limit.
    CFG.debug = True
    CFG.debug_sample_size = 500
    CFG.p1_epochs = 2
    CFG.p2_epochs = 1

    # 1. Train
    run_training(logger)

    # 2. Validate
    acc, probs, targets, val_df = run_holdout_validation(logger)

    # 3. Failure Analysis
    if probs is not None:
        run_failure_analysis(probs, targets, val_df, logger)

    # 4. Submission
    # Threshold check as per requirements
    threshold = 0.9076
    if acc > threshold:
        logger.info(f"Validation accuracy {acc} > {threshold}. Generating submission.")
        inference()
    else:
        logger.info(
            f"Validation accuracy {acc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
