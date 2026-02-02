import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Ensure library is in path
sys.path.append(".")

# Import Library Modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_data, ISICDataset, get_transforms
from library.train_engine import run_fold
from library.inference import generate_submission, predict_with_tta
from library.model import HybridEfficientNet


def main():
    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # ---------------------------------------------------------
    # Adjust parameters to ensure execution finishes within 23 minutes
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 2  # Train only 2 folds for the baseline check
    Config.DEBUG = True

    # Setup Logger and Seed
    logger = get_logger("Main")
    seed_everything(Config.SEED)

    logger.info("Starting Fast Baseline Run...")

    # ---------------------------------------------------------
    # 2. Data Processing & Subsampling
    # ---------------------------------------------------------
    logger.info("Processing Data...")
    # Generate full processed data first
    df_train_full, df_test, feature_cols = process_data(load_cached_data=False)

    # Subsample to speed up training
    # Target size: ~10,000 samples (Large enough for signal, small enough for speed)
    TARGET_SIZE = 10000

    logger.info(f"Original Training Data Shape: {df_train_full.shape}")

    df_pos = df_train_full[df_train_full["target"] == 1]
    df_neg = df_train_full[df_train_full["target"] == 0]

    # Keep all positives, sample negatives
    n_pos = len(df_pos)
    n_neg_keep = TARGET_SIZE - n_pos

    if len(df_neg) > n_neg_keep:
        df_neg_sampled = df_neg.sample(n=n_neg_keep, random_state=Config.SEED)
    else:
        df_neg_sampled = df_neg

    # Combine and shuffle
    df_train_subsampled = (
        pd.concat([df_pos, df_neg_sampled])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    # Re-assign folds using StratifiedGroupKFold to maintain patient independence
    # We must re-do this because random subsampling destroys the validity of the original fold column
    sgkf = StratifiedGroupKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    df_train_subsampled["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            df_train_subsampled,
            df_train_subsampled["target"],
            df_train_subsampled["patient_id"],
        )
    ):
        df_train_subsampled.loc[val_idx, "fold"] = fold

    logger.info(f"Subsampled Training Data Shape: {df_train_subsampled.shape}")

    # Overwrite the cached parquet file so run_fold picks up the subsampled data
    train_cache_path = os.path.join(Config.WORKING_DIR, "processed_train.parquet")
    df_train_subsampled.to_parquet(train_cache_path, index=False)

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    logger.info("Starting Training Loop...")
    for fold in range(Config.NUM_FOLDS):
        run_fold(fold, load_cached_data=True)

    # ---------------------------------------------------------
    # 4. Validation on Hold-Out Set
    # ---------------------------------------------------------
    logger.info("Performing Validation on Hold-Out Set...")

    # Load the strict hold-out metadata
    df_val_holdout = pd.read_csv(Config.VAL_METADATA_PATH)

    # Retrieve the preprocessed features for these hold-out images from the full dataset
    # (df_train_full contains processed features for both train and val sets)
    holdout_images = set(df_val_holdout["image_name"])
    df_holdout_processed = df_train_full[
        df_train_full["image_name"].isin(holdout_images)
    ].reset_index(drop=True)

    if len(df_holdout_processed) == 0:
        logger.error("No matched holdout images found in processed data.")
        return

    # Create Dataset and Loader
    val_dataset = ISICDataset(
        df_holdout_processed,
        transforms=get_transforms(mode="valid"),
        mode="test",  # 'test' mode avoids expecting targets in __getitem__ return dict
        tabular_cols=feature_cols,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Ensemble Inference
    device = torch.device(Config.DEVICE)
    ensemble_preds = {}

    for fold in range(Config.NUM_FOLDS):
        # Initialize Model
        model = HybridEfficientNet(
            model_name=Config.MODEL_NAME,
            pretrained=False,
            num_classes=Config.NUM_CLASSES,
            num_tabular_features=len(feature_cols),
            tabular_hidden_dim=Config.TABULAR_HIDDEN_DIM,
            final_dropout=Config.FINAL_DROPOUT,
        )
        model.to(device)

        # Load Checkpoint
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_best.pth")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)

            # Predict
            fold_preds = predict_with_tta(model, val_loader, device)

            # Aggregate
            for img, prob in fold_preds.items():
                if img not in ensemble_preds:
                    ensemble_preds[img] = 0.0
                ensemble_preds[img] += prob
        else:
            logger.warning(f"Checkpoint for fold {fold} not found.")

    # Average Predictions
    final_preds = []
    targets = []

    # Create a map for ground truth
    img_to_target = dict(
        zip(df_holdout_processed["image_name"], df_holdout_processed["target"])
    )

    pred_list = []
    target_list = []

    # Ensure alignment
    for img in ensemble_preds.keys():
        avg_prob = ensemble_preds[img] / Config.NUM_FOLDS
        pred_list.append(avg_prob)
        target_list.append(img_to_target[img])

    # Compute Metric
    auc = roc_auc_score(target_list, pred_list)
    print(f"Final Validation Metric: {auc}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Prepare analysis dataframe
    # We merge predictions back to the original metadata to get raw features (age, sex, site)
    df_preds = pd.DataFrame(
        {"image_name": list(ensemble_preds.keys()), "pred": pred_list}
    )
    df_analysis = pd.merge(df_val_holdout, df_preds, on="image_name")

    # Calculate Error
    df_analysis["error"] = (df_analysis["target"] - df_analysis["pred"]).abs()

    # Correlation with Age
    corr_age = df_analysis["age_approx"].corr(df_analysis["error"])
    print(f"Correlation Error vs Age: {corr_age}")

    # Correlation with Sex
    if "sex" in df_analysis.columns:
        # Simple encoding for correlation
        df_analysis["sex_enc"] = df_analysis["sex"].astype("category").cat.codes
        corr_sex = df_analysis["sex_enc"].corr(df_analysis["error"])
        print(f"Correlation Error vs Sex: {corr_sex}")

    # Correlation with Anatomical Site
    if "anatom_site_general_challenge" in df_analysis.columns:
        df_analysis["site_enc"] = (
            df_analysis["anatom_site_general_challenge"].astype("category").cat.codes
        )
        corr_site = df_analysis["site_enc"].corr(df_analysis["error"])
        print(f"Correlation Error vs Site: {corr_site}")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9094590472584224

    if auc > THRESHOLD:
        logger.info(
            f"Validation metric ({auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        logger.info(
            f"Validation metric ({auc}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
