import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import cv2
from torch.cuda.amp import GradScaler

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    worker_init_fn,
    calculate_pos_weights,
)
from library.dataset import get_data, AppleDataset
from library.model import AppleDiseaseModel
from library.loss import WeightedBCELoss
from library.train import train_one_epoch, validate
from library.inference import predict_tta, rank_average, reconstruct_probabilities


def run_failure_analysis(oof_df, logger):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    logger.info("Starting Failure Analysis...")

    # Calculate Error Magnitude (MSE per sample across 4 classes)
    # Targets
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    targets = oof_df[target_cols].values

    # Predictions
    pred_cols = [f"pred_{c}" for c in target_cols]
    preds = oof_df[pred_cols].values

    # MSE per sample
    errors = np.mean((targets - preds) ** 2, axis=1)
    oof_df["error_magnitude"] = errors

    # Extract Meta Features
    # We need to read image files to get width/height/size
    # This might be slow, so we'll do it for the validation set only
    file_sizes = []
    widths = []
    heights = []

    for _, row in oof_df.iterrows():
        path = row["full_path"]
        if os.path.exists(path):
            file_sizes.append(os.path.getsize(path))
            # Read image for dims
            img = cv2.imread(path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    oof_df["file_size"] = file_sizes
    oof_df["width"] = widths
    oof_df["height"] = heights

    # Calculate Correlations
    correlations = {}
    for feature in ["file_size", "width", "height"]:
        if oof_df[feature].std() > 0:
            corr = oof_df["error_magnitude"].corr(oof_df[feature])
            correlations[feature] = corr
        else:
            correlations[feature] = 0.0

    logger.info("Correlation between Error Magnitude and Input Features:")
    for feature, corr in correlations.items():
        logger.info(f"  {feature}: {corr:.4f}")

    return correlations


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 10  # Reduced epochs for speed
    Config.USE_SWA = False  # Disable SWA for speed

    seed_everything(Config.SEED)
    logger = get_logger("runfile")

    # 2. Data Preparation
    logger.info("Loading Data...")
    train_df_part = get_data("train")
    val_df_part = get_data("val")
    full_df = pd.concat([train_df_part, val_df_part], axis=0).reset_index(drop=True)

    # Container for OOF predictions
    # We need to store predictions for each model to ensemble them later
    # Structure: {model_name: DataFrame with image_id and pred_rust, pred_scab}
    model_oof_preds = {}

    # 3. Training Loop (Ensemble)
    for model_cfg in Config.MODELS:
        model_name = model_cfg["name"]
        safe_model_name = model_name.replace(".", "_")
        logger.info(f"Processing Model: {model_name}")

        # Initialize OOF storage for this model
        # We will fill this with probabilities for the 2 binary targets
        oof_preds = np.zeros((len(full_df), 2))  # [Rust, Scab]

        skf = StratifiedKFold(
            n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_df, full_df["stratify_label"])
        ):
            logger.info(f"  Fold {fold + 1}/{Config.FOLDS}")

            # Split
            train_sub = full_df.iloc[train_idx].reset_index(drop=True)
            val_sub = full_df.iloc[val_idx].reset_index(drop=True)

            # Loaders
            train_ds = AppleDataset(
                train_sub, img_size=model_cfg["img_size"], mode="train"
            )
            val_ds = AppleDataset(val_sub, img_size=model_cfg["img_size"], mode="val")

            # Handle small batches
            drop_last = len(train_ds) >= model_cfg["batch_size"]

            train_loader = DataLoader(
                train_ds,
                batch_size=model_cfg["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                worker_init_fn=worker_init_fn,
                drop_last=drop_last,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model & Training Components
            pos_weights = calculate_pos_weights(train_sub).to(Config.DEVICE)

            model = AppleDiseaseModel(
                model_name=model_name,
                pretrained=True,
                num_classes=Config.NUM_TARGETS,
                gem_p=model_cfg["gem_p"],
                num_msd=model_cfg["num_msd"],
                msd_dropout=model_cfg["msd_dropout"],
            ).to(Config.DEVICE)

            criterion = WeightedBCELoss(
                pos_weights=pos_weights, smoothing=Config.LABEL_SMOOTHING
            )
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=1e-6
            )

            # Scaler for AMP
            scaler = GradScaler(enabled=Config.USE_AMP)

            best_auc = 0.0
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold}.pth"
            )

            # Training Epochs
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    Config.DEVICE,
                    epoch,
                    logger,
                    scaler=scaler,
                    accumulation_steps=Config.GRAD_ACCUM_STEPS,
                )
                scheduler.step()

                val_auc, val_loss = validate(
                    val_loader, model, criterion, Config.DEVICE
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # Load best model for OOF generation
            model.load_state_dict(
                torch.load(best_model_path, map_location=Config.DEVICE)
            )
            model.eval()

            # Generate OOF predictions
            fold_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(Config.DEVICE)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)
                    fold_preds.append(preds.cpu().numpy())

            fold_preds = np.concatenate(fold_preds, axis=0)
            oof_preds[val_idx] = fold_preds

            del model, optimizer, scheduler, criterion
            torch.cuda.empty_cache()

        model_oof_preds[model_name] = oof_preds

    # 4. Ensemble & Metric Calculation
    logger.info("Calculating Ensemble Metrics...")

    # Collect predictions list for Rank Averaging
    pred_list = list(model_oof_preds.values())

    # Rank Average
    avg_ranks = rank_average(pred_list)

    # Reconstruct 4-class probabilities
    final_oof_probs = reconstruct_probabilities(avg_ranks)

    # Calculate Metric
    # Ground Truth
    gt_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    gt_values = full_df[gt_cols].values

    auc_scores = []
    for i, col in enumerate(gt_cols):
        try:
            score = roc_auc_score(gt_values[:, i], final_oof_probs[:, i])
            auc_scores.append(score)
        except ValueError:
            auc_scores.append(0.5)  # Handle single class edge case

    final_metric = np.mean(auc_scores)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Create DataFrame for analysis
    analysis_df = full_df.copy()
    for i, col in enumerate(gt_cols):
        analysis_df[f"pred_{col}"] = final_oof_probs[:, i]

    run_failure_analysis(analysis_df, logger)

    # 6. Submission Logic
    THRESHOLD = 0.9954104122251848

    if final_metric > THRESHOLD:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_df = get_data("test")
        all_model_test_preds = []

        for model_cfg in Config.MODELS:
            model_name = model_cfg["name"]
            safe_model_name = model_name.replace(".", "_")
            img_size = model_cfg["img_size"]

            test_ds = AppleDataset(test_df, img_size=img_size, mode="test")
            test_loader = DataLoader(
                test_ds,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Average across folds
            fold_preds_accum = np.zeros((len(test_df), Config.NUM_TARGETS))

            for fold in range(Config.FOLDS):
                ckpt_path = os.path.join(
                    Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold}.pth"
                )

                model = AppleDiseaseModel(
                    model_name=model_name,
                    pretrained=False,
                    num_classes=Config.NUM_TARGETS,
                    gem_p=model_cfg["gem_p"],
                    num_msd=model_cfg["num_msd"],
                    msd_dropout=model_cfg["msd_dropout"],
                ).to(Config.DEVICE)

                model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))

                # Predict with TTA
                preds = predict_tta(model, test_loader, Config.DEVICE)
                fold_preds_accum += preds

                del model
                torch.cuda.empty_cache()

            # Average folds
            avg_fold_preds = fold_preds_accum / Config.FOLDS
            all_model_test_preds.append(avg_fold_preds)

        # Rank Average Ensemble
        final_test_ranks = rank_average(all_model_test_preds)
        final_test_probs = reconstruct_probabilities(final_test_ranks)

        # Save
        sub_df = pd.DataFrame(
            {
                "image_id": test_df["image_id"],
                "healthy": final_test_probs[:, 0],
                "multiple_diseases": final_test_probs[:, 1],
                "rust": final_test_probs[:, 2],
                "scab": final_test_probs[:, 3],
            }
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
