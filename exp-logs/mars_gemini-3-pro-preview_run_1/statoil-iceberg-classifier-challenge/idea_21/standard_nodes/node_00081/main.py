import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import log_loss
import logging

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import network
from library import engine
from library import inference
from library import pseudo_labeling

# =============================================================================
# SETUP & CONSTANTS
# =============================================================================
logger = utils.setup_logger("idea_21", os.path.join(config.WORK_DIR, "execution.log"))
DEVICE = utils.get_device()
THRESHOLD_METRIC = 0.16918645240183008


def main():
    utils.set_seed(config.SEED)
    logger.info(f"Using device: {DEVICE}")

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    logger.info("Loading data splits...")
    train_data_dict, val_data_dict, test_data_dict = dataset.load_data_splits(
        load_cached_data=True
    )

    # Create Datasets
    # Training set (from train_metadata)
    train_dataset = dataset.IcebergDataset(
        images=train_data_dict["images"],
        angles=train_data_dict["angles"],
        labels=train_data_dict["labels"],
        ids=train_data_dict["ids"],
        transform=dataset.get_transforms(phase="train"),
    )

    # Validation set (from val_metadata - HOLD OUT)
    val_dataset = dataset.IcebergDataset(
        images=val_data_dict["images"],
        angles=val_data_dict["angles"],
        labels=val_data_dict["labels"],
        ids=val_data_dict["ids"],
        transform=dataset.get_transforms(phase="val"),
    )

    # Test set (for pseudo-labeling and submission)
    test_dataset = dataset.IcebergDataset(
        images=test_data_dict["images"],
        angles=test_data_dict["angles"],
        labels=None,
        ids=test_data_dict["ids"],
        transform=dataset.get_transforms(phase="test"),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
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

    # =========================================================================
    # 2. STAGE 1: TEACHER ENSEMBLE TRAINING
    # =========================================================================
    logger.info("--- Starting Stage 1: Teacher Ensemble Training ---")

    teacher_model_paths = []
    num_teachers = 5  # As per idea description

    # Directory for checkpoints
    ckpt_dir = os.path.join(config.WORK_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for i in range(num_teachers):
        logger.info(f"Training Teacher Model {i+1}/{num_teachers}")

        # Seed variation for ensemble diversity
        utils.set_seed(config.SEED + i)

        model = network.IcebergResNet().to(DEVICE)
        optimizer, scheduler = engine.get_optimizer_scheduler(model)

        # SWA Setup
        swa_model = engine.get_swa_model(model)
        swa_start_epoch = config.MAX_EPOCHS - config.SWA_EPOCHS

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, config.MAX_EPOCHS + 1):
            # Train
            train_loss = engine.train_one_epoch(
                model, train_loader, optimizer, DEVICE, epoch
            )

            # SWA Update
            if epoch > swa_start_epoch:
                swa_model.update_parameters(model)

            # Validation (for scheduler and early stopping monitoring)
            val_loss, _, _ = engine.evaluate(model, val_loader, DEVICE)

            # Scheduler Step
            scheduler.step(val_loss)

            # Early Stopping Check (on standard model)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (
                patience_counter >= config.EARLY_STOPPING_PATIENCE
                and epoch < swa_start_epoch
            ):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                # If we stop early, we might not have reached SWA phase.
                # In this fast baseline, we ensure we run at least until SWA starts or handle it.
                # For robustness, we'll continue to SWA phase if close, or just break.
                # Given strict SWA requirement, we force running SWA at the end.
                # To keep it simple: we rely on fixed epochs for SWA part.
                pass

        # Finalize SWA
        engine.update_swa_bn(train_loader, swa_model, DEVICE)

        # Save Teacher
        save_path = os.path.join(ckpt_dir, f"teacher_{i}.pth")
        utils.save_checkpoint(
            {"model_state_dict": swa_model.module.state_dict()},
            is_best=False,
            checkpoint_dir=ckpt_dir,
            filename=f"teacher_{i}.pth",
        )
        teacher_model_paths.append(save_path)

    # =========================================================================
    # 3. STAGE 2: PSEUDO-LABELING
    # =========================================================================
    logger.info("--- Starting Stage 2: Pseudo-Labeling ---")

    # Generate stats
    stats_df = pseudo_labeling.generate_ensemble_stats(
        teacher_model_paths, test_loader, DEVICE, load_cached_data=False
    )

    # Filter
    pseudo_labels_df = pseudo_labeling.filter_pseudo_labels(stats_df)

    # Extract Data
    p_imgs, p_angs, p_lbls, p_ids = pseudo_labeling.extract_pseudo_dataset(
        test_data_dict, pseudo_labels_df
    )

    if len(p_imgs) > 0:
        # Create Combined Dataset
        logger.info(
            f"Augmenting training set with {len(p_imgs)} pseudo-labeled samples."
        )

        # Combine arrays
        combined_imgs = np.concatenate([train_data_dict["images"], p_imgs], axis=0)
        combined_angs = np.concatenate([train_data_dict["angles"], p_angs], axis=0)
        combined_lbls = np.concatenate([train_data_dict["labels"], p_lbls], axis=0)
        combined_ids = np.concatenate([train_data_dict["ids"], p_ids], axis=0)

        combined_dataset = dataset.IcebergDataset(
            images=combined_imgs,
            angles=combined_angs,
            labels=combined_lbls,
            ids=combined_ids,
            transform=dataset.get_transforms(phase="train"),
        )

        combined_loader = DataLoader(
            combined_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        logger.warning(
            "No pseudo-labels generated. Proceeding with original training set."
        )
        combined_loader = train_loader

    # =========================================================================
    # 4. STAGE 3: STUDENT ENSEMBLE TRAINING
    # =========================================================================
    logger.info("--- Starting Stage 3: Student Ensemble Training ---")

    student_model_paths = []
    num_students = 5

    # Scale epochs based on data size increase?
    # For simplicity/safety in this baseline, we keep config epochs but rely on scheduler.

    for i in range(num_students):
        logger.info(f"Training Student Model {i+1}/{num_students}")
        utils.set_seed(config.SEED + 100 + i)  # Different seeds

        model = network.IcebergResNet().to(DEVICE)
        optimizer, scheduler = engine.get_optimizer_scheduler(model)

        swa_model = engine.get_swa_model(model)
        swa_start_epoch = config.MAX_EPOCHS - config.SWA_EPOCHS

        for epoch in range(1, config.MAX_EPOCHS + 1):
            engine.train_one_epoch(model, combined_loader, optimizer, DEVICE, epoch)

            if epoch > swa_start_epoch:
                swa_model.update_parameters(model)

            # Validate on original validation set
            val_loss, _, _ = engine.evaluate(model, val_loader, DEVICE)
            scheduler.step(val_loss)

        engine.update_swa_bn(combined_loader, swa_model, DEVICE)

        save_path = os.path.join(ckpt_dir, f"student_{i}.pth")
        utils.save_checkpoint(
            {"model_state_dict": swa_model.module.state_dict()},
            is_best=False,
            checkpoint_dir=ckpt_dir,
            filename=f"student_{i}.pth",
        )
        student_model_paths.append(save_path)

    # =========================================================================
    # 5. FINAL EVALUATION & FAILURE ANALYSIS
    # =========================================================================
    logger.info("--- Final Evaluation ---")

    # Predict on Validation Set using Student Ensemble
    df_val_preds = inference.ensemble_predict(
        student_model_paths, val_loader, DEVICE, output_path=None
    )

    # Merge with ground truth
    # val_data_dict['ids'] and val_data_dict['labels'] are aligned with val_loader order
    # But ensemble_predict returns a DF aligned with loader iteration.
    # We can trust the ID matching.

    gt_df = pd.DataFrame(
        {
            "id": val_data_dict["ids"],
            "label": val_data_dict["labels"],
            "inc_angle": val_data_dict["angles"],
        }
    )

    # Merge
    eval_df = pd.merge(gt_df, df_val_preds, on="id")

    # Calculate Metric
    y_true = eval_df["label"].values
    y_pred = eval_df["is_iceberg"].values

    final_metric = log_loss(y_true, y_pred, labels=[0, 1])

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate per-sample error (log loss contribution or absolute error)
    # Using absolute error for correlation analysis as it's more intuitive for "magnitude"
    eval_df["abs_error"] = np.abs(eval_df["label"] - eval_df["is_iceberg"])

    # Calculate simple image stats for correlation
    # We'll do this on the fly for the validation set
    val_imgs = val_data_dict["images"]
    b1_means = []
    b2_means = []
    for i in range(len(val_imgs)):
        # Raw images are (75,75,2)
        img = val_imgs[i]
        b1_means.append(np.mean(img[:, :, 0]))
        b2_means.append(np.mean(img[:, :, 1]))

    eval_df["b1_mean"] = b1_means
    eval_df["b2_mean"] = b2_means

    # Correlations
    corr_angle = eval_df["abs_error"].corr(eval_df["inc_angle"])
    corr_b1 = eval_df["abs_error"].corr(eval_df["b1_mean"])
    corr_b2 = eval_df["abs_error"].corr(eval_df["b2_mean"])

    print("Failure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Band 1 Mean: {corr_b1:.4f}")
    print(f"  Band 2 Mean: {corr_b2:.4f}")

    # =========================================================================
    # 6. SUBMISSION
    # =========================================================================
    if final_metric < THRESHOLD_METRIC:
        logger.info(
            f"Validation metric {final_metric:.6f} < {THRESHOLD_METRIC}. Generating submission..."
        )
        inference.ensemble_predict(
            student_model_paths, test_loader, DEVICE, output_path=config.SUBMISSION_PATH
        )
    else:
        logger.warning(
            f"Validation metric {final_metric:.6f} >= {THRESHOLD_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
