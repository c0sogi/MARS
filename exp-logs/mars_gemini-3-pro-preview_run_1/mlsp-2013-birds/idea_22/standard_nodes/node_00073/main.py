import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config, utils, data, model, training, inference


def run_failure_analysis(val_probs, val_targets, val_ids):
    """
    Analyzes prediction errors on the validation set and correlates them with
    supplemental histogram features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Mean Absolute Error per sample (N,)
    mae_per_sample = np.mean(np.abs(val_probs - val_targets), axis=1)

    # Load Histogram Features
    hist_path = os.path.join(
        config.INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    if not os.path.exists(hist_path):
        print("Histogram features not found. Skipping feature correlation analysis.")
        return

    try:
        # Read histogram features
        # The file is CSV-like. Based on EDA, it likely has a header or we infer it.
        # We assume standard CSV reading works.
        df_hist = pd.read_csv(hist_path)

        # Ensure first column is 'rec_id'
        if "rec_id" not in df_hist.columns:
            df_hist.rename(columns={df_hist.columns[0]: "rec_id"}, inplace=True)

        # Create analysis dataframe
        df_analysis = pd.DataFrame({"rec_id": val_ids, "error": mae_per_sample})

        # Merge with features
        df_merged = pd.merge(df_analysis, df_hist, on="rec_id", how="inner")

        if len(df_merged) == 0:
            print("No matching records found for failure analysis.")
            return

        # Calculate correlations
        feature_cols = [c for c in df_merged.columns if c not in ["rec_id", "error"]]

        correlations = {}
        for col in feature_cols:
            # Only correlate numeric columns
            if pd.api.types.is_numeric_dtype(df_merged[col]):
                corr = df_merged["error"].corr(df_merged[col])
                if not np.isnan(corr):
                    correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Top 5 Features correlated with Error Magnitude:")
        for name, val in sorted_corr[:5]:
            print(f"  {name}: {val:.4f}")

    except Exception as e:
        print(f"Failure analysis failed: {e}")


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    logger = utils.get_logger("Runfile")
    device = config.DEVICE

    logger.info(f"Running on device: {device}")

    # =========================================================================
    # STAGE 1: TRAIN TEACHERS
    # =========================================================================
    logger.info("=== Stage 1: Training Teachers ===")

    teacher_paths = []

    # Train 3 Independent Teachers
    for i in range(3):
        logger.info(f"--- Training Teacher {i} ---")

        # Vary seed slightly for independence in initialization/batching
        utils.set_seed(config.SEED + i)

        # Get DataLoaders (Standard Labeled Train)
        train_loader, val_loader, _ = data.get_dataloaders(
            use_pseudo_labels=False, load_cached_data=True
        )

        # Initialize Model
        net = model.get_model(device=device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

        # Checkpoint Directory
        ckpt_dir = os.path.join(config.WORKING_DIR, f"teacher_{i}")

        # Trainer
        trainer = training.Trainer(
            model=net,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        # Fit
        trainer.fit(config.EPOCHS)

        # Store SWA model path
        teacher_paths.append(os.path.join(ckpt_dir, "model_swa.pth"))

        # Clean up to free memory
        del net, optimizer, scheduler, trainer, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: PSEUDO-LABEL GENERATION
    # =========================================================================
    logger.info("=== Stage 2: Generating Pseudo-Labels ===")

    # Load Teachers
    teachers = []
    for path in teacher_paths:
        net = model.get_model(device=device, weights_path=path)
        net.eval()
        teachers.append(net)

    # Generate Ensemble Predictions
    # This averages predictions across all teachers (Fixed Resolution, No TTA)
    # Cite Lesson 00072: Preserving Spectrotemporal Geometry
    pseudo_probs = inference.predict_ensemble(teachers, device)

    # Save Pseudo-Labels
    inference.save_pseudo_labels(pseudo_probs, config.PSEUDO_LABELS_PATH)

    # Clean up
    del teachers
    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: TRAIN STUDENT
    # =========================================================================
    logger.info("=== Stage 3: Training Student ===")

    # Reset seed for Student training
    utils.set_seed(config.SEED + 100)

    # Get DataLoaders (Combined Train + Pseudo-Labeled Test)
    train_loader, val_loader, _ = data.get_dataloaders(
        use_pseudo_labels=True, load_cached_data=True
    )

    # Initialize Student Model
    student_net = model.get_model(device=device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        student_net.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Checkpoint Directory
    ckpt_dir = os.path.join(config.WORKING_DIR, "student")

    # Trainer
    trainer = training.Trainer(
        model=student_net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=ckpt_dir,
    )

    # Fit
    trainer.fit(config.EPOCHS)

    # Save final student SWA model to the expected location
    student_swa_path = os.path.join(ckpt_dir, "model_swa.pth")
    shutil.copy(student_swa_path, config.STUDENT_CHECKPOINT_PATH)

    # Clean up
    del student_net, optimizer, scheduler, trainer, train_loader
    torch.cuda.empty_cache()

    # =========================================================================
    # VALIDATION & ANALYSIS
    # =========================================================================
    logger.info("=== Evaluation ===")

    # Load Final Student SWA Model
    final_model = model.get_model(
        device=device, weights_path=config.STUDENT_CHECKPOINT_PATH
    )
    final_model.eval()

    # Get Val Loader (Standard, no pseudo labels)
    _, val_loader, _ = data.get_dataloaders(
        use_pseudo_labels=False, load_cached_data=True
    )

    # Predict on Validation Set
    val_probs = inference.predict_probs(
        final_model, val_loader, device, apply_flip=False
    )

    # Retrieve Validation Targets and IDs
    df_val = data.load_metadata("val", load_cached_data=True)

    # Respect debug limit if set
    if config.DEBUG_MAX_SAMPLES is not None:
        df_val = df_val.iloc[: config.DEBUG_MAX_SAMPLES]

    val_ids = df_val["rec_id"].values
    label_cols = [f"species_{i}" for i in range(config.NUM_CLASSES)]
    val_targets = df_val[label_cols].values

    # Calculate ROC AUC
    aucs = []
    for i in range(config.NUM_CLASSES):
        # Only calculate AUC if both classes are present in the subset
        if len(np.unique(val_targets[:, i])) > 1:
            score = roc_auc_score(val_targets[:, i], val_probs[:, i])
            aucs.append(score)

    final_metric = np.mean(aucs) if aucs else 0.5

    # Print Metric in required format
    print(f"Final Validation Metric: {final_metric}")

    # Run Failure Analysis
    run_failure_analysis(val_probs, val_targets, val_ids)

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    threshold = 0.9594082190886809

    if final_metric > threshold:
        logger.info("Metric threshold met. Generating submission...")
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        inference.generate_submission(final_model, device, sub_path)
    else:
        logger.info(
            f"Metric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
