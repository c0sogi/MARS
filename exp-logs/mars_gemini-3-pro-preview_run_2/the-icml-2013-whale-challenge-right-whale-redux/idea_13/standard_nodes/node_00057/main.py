import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import WhaleClassifier
from library.engine import train_one_epoch, validate
from library.stacking import StackingMetaLearner, generate_submission


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "train.log"))

    # Override Config for Fast Baseline
    # Increased to 12 epochs as we reduced ensemble size, allowing for better convergence (Cite solution_lesson_node_00055)
    Config.EPOCHS = 12
    logger.info(f"Setting EPOCHS to {Config.EPOCHS} for execution.")

    # 2. Prepare Metadata and Split Logic
    # Reconstruct the full training set to map OOF predictions correctly.
    train_df_orig = pd.read_csv(Config.TRAIN_CSV)
    val_df_orig = pd.read_csv(Config.VAL_CSV)
    full_train_df = pd.concat([train_df_orig, val_df_orig], ignore_index=True)

    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        full_train_df = full_train_df.iloc[: Config.DEBUG_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SIZE]
        logger.info(f"DEBUG MODE: Using {len(full_train_df)} train samples.")

    # Replicate the split logic to get indices for OOF mapping
    y_full = full_train_df["label"].values
    # Dummy X just for the split generator
    X_dummy = np.zeros(len(y_full))

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X_dummy, y_full))

    # Storage for OOF and Test Predictions
    # model_oof_preds: Dictionary {model_name: array of shape (N_samples,)}
    # model_test_preds: Dictionary {model_name: array of shape (N_test,)}
    model_oof_preds = {}
    model_test_preds = {}

    # 3. Iterate over Base Models
    for model_cfg in Config.BASE_MODELS:
        model_name = model_cfg["name"]
        arch = model_cfg["arch"]
        hop_length = model_cfg["hop_length"]
        in_channels = model_cfg["in_channels"]

        logger.info(
            f"\n=== Processing Model: {model_name} (Arch: {arch}, Hop: {hop_length}) ==="
        )

        # Initialize storage for this model
        this_model_oof = np.zeros(len(full_train_df))
        this_model_test_accum = np.zeros(len(test_df))

        # Iterate over Folds
        for fold in range(Config.NUM_FOLDS):
            logger.info(f"  Fold {fold}/{Config.NUM_FOLDS - 1}")

            # Get DataLoaders (handles caching internally)
            train_loader, val_loader, test_loader = get_dataloaders(
                fold=fold,
                hop_length=hop_length,
                load_cached_data=True,
                batch_size=Config.BATCH_SIZE,
            )

            # Initialize Model
            model = WhaleClassifier(
                model_name=arch, pretrained=True, in_chans=in_channels, num_classes=1
            )
            model = model.to(Config.DEVICE)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Training Loop
            best_val_loss = float("inf")
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, Config.DEVICE
                )
                val_loss, val_auc = validate(model, val_loader, Config.DEVICE)
                scheduler.step()

                # Save best model based on Val Loss (Calibration focus)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)

            # Load Best Model for Inference
            model.load_state_dict(
                torch.load(best_model_path, map_location=Config.DEVICE)
            )
            model.eval()

            # Predict on Validation (OOF)
            # Retrieve validation indices for this fold
            _, val_idx = splits[fold]

            val_probs = []
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(Config.DEVICE)
                    output = model(data)
                    probs = torch.sigmoid(output).cpu().numpy().flatten()
                    val_probs.append(probs)
            val_probs = np.concatenate(val_probs)

            # Store OOF predictions
            this_model_oof[val_idx] = val_probs

            # Predict on Test
            test_probs = []
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(Config.DEVICE)
                    output = model(data)
                    probs = torch.sigmoid(output).cpu().numpy().flatten()
                    test_probs.append(probs)
            test_probs = np.concatenate(test_probs)

            this_model_test_accum += test_probs

            # Cleanup to free GPU memory
            del model, optimizer, scheduler, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()

        # Store aggregated results for this model type
        model_oof_preds[model_name] = this_model_oof
        model_test_preds[model_name] = this_model_test_accum / Config.NUM_FOLDS

        # Log individual model performance
        model_auc = roc_auc_score(y_full, this_model_oof)
        logger.info(f"Model {model_name} OOF AUC: {model_auc:.5f}")

    # 4. Stacking
    logger.info("\n=== Training Stacking Meta-Learner ===")

    # Prepare feature matrices
    # Sort keys to ensure consistent column ordering
    sorted_keys = sorted(model_oof_preds.keys())
    X_oof = np.column_stack([model_oof_preds[k] for k in sorted_keys])
    X_test = np.column_stack([model_test_preds[k] for k in sorted_keys])

    meta_learner = StackingMetaLearner()
    # Fit returns the AUC on the OOF set
    final_oof_auc = meta_learner.fit(X_oof, y_full)

    # Generate final test probabilities
    final_test_probs = meta_learner.predict(X_test)

    # Print Required Metric
    print(f"Final Validation Metric: {final_oof_auc}")

    # 5. Failure Analysis
    logger.info("\n=== Failure Analysis ===")

    # Get calibrated OOF predictions from the meta-learner
    oof_preds_final = meta_learner.model.predict_proba(X_oof)[:, 1]
    errors = np.abs(y_full - oof_preds_final)

    # Extract audio duration for correlation analysis
    logger.info("Extracting audio features for failure analysis...")
    durations = []

    for idx, row in full_train_df.iterrows():
        path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            # sf.info is fast and reads header only
            info = sf.info(path)
            durations.append(info.duration)
        except Exception as e:
            durations.append(0.0)

    durations = np.array(durations)

    # Calculate correlations
    if len(np.unique(durations)) > 1:
        corr_dur, _ = pearsonr(errors, durations)
        print(f"Correlation between Error and Duration: {corr_dur:.4f}")
    else:
        print("Correlation between Error and Duration: Undefined (constant duration)")

    corr_label, _ = pearsonr(errors, y_full)
    print(f"Correlation between Error and Target Label: {corr_label:.4f}")

    # 6. Submission
    threshold = 0.9959928858461402
    if final_oof_auc > threshold:
        logger.info(
            f"Validation metric {final_oof_auc} exceeds threshold {threshold}. Generating submission."
        )
        generate_submission(final_test_probs)
    else:
        logger.warning(
            f"Validation metric {final_oof_auc} DOES NOT exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
