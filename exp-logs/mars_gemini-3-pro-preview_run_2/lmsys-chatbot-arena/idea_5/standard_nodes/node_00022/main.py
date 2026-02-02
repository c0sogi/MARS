import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from transformers import AutoTokenizer

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_data, get_folds, ChatbotDataset
from library.trainer import run_fold
from library.inference import generate_submission
from library.model import SiameseModel


def main():
    # 1. Setup and Configuration Overrides for Fast Baseline
    seed_everything(Config.SEED)
    logger = get_logger("main")

    # Override Config for speed and resource utilization
    # We use 2 folds and 1 epoch with subsampling to ensure < 2 hours runtime
    Config.EPOCHS = 1
    Config.N_FOLDS = 2
    Config.TRAIN_BATCH_SIZE = 16
    Config.VALID_BATCH_SIZE = 32

    # Subsampling size for training to ensure quick baseline execution
    SAMPLE_SIZE = 5000

    logger.info("Configuration set for fast baseline execution.")

    # 2. Data Loading
    # load_data handles caching, feature engineering, and normalization
    train_df_full, test_df = load_data(load_cached_data=True)

    # Apply Stratified K-Fold splitting
    train_df_full = get_folds(train_df_full, n_folds=Config.N_FOLDS)

    # 3. Training Loop
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    for fold_idx in range(Config.N_FOLDS):
        logger.info(f"Preparing data for Fold {fold_idx}...")

        # Split data into train/val for this fold
        train_fold = train_df_full[train_df_full["fold"] != fold_idx].copy()
        val_fold = train_df_full[train_df_full["fold"] == fold_idx].copy()

        # Subsample training data for speed
        if len(train_fold) > SAMPLE_SIZE:
            logger.info(
                f"Subsampling training data from {len(train_fold)} to {SAMPLE_SIZE} samples."
            )
            train_fold = train_fold.sample(
                n=SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

        # Run training pipeline for this fold
        run_fold(fold_idx, train_fold, val_fold, tokenizer)

    # 4. Validation Assessment
    logger.info("Starting Validation Assessment on Hold-out Set...")

    # Load hold-out validation IDs from metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        logger.error(f"Validation metadata not found at {val_meta_path}")
        return

    val_meta_df = pd.read_csv(val_meta_path)
    val_ids = val_meta_df["id"].values

    # Filter the processed dataframe (which has normalized features) to get the validation set
    val_processed = (
        train_df_full[train_df_full["id"].isin(val_ids)].copy().reset_index(drop=True)
    )

    if len(val_processed) == 0:
        logger.error("Validation set is empty after filtering. Check IDs.")
        return

    # Prepare DataLoader for Validation Inference
    device = torch.device(Config.DEVICE)

    # We use is_test=True to skip internal target processing in __getitem__
    # (though we have targets in the df, we handle them manually for metric calc)
    val_dataset = ChatbotDataset(
        val_processed, tokenizer, max_length=Config.MAX_LENGTH, is_test=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference on Validation Set
    ensemble_probs = np.zeros((len(val_processed), 3), dtype=np.float32)
    models_found = 0

    for fold_idx in range(Config.N_FOLDS):
        model_path = os.path.join(
            Config.MODEL_OUTPUT_DIR, f"best_model_fold_{fold_idx}.pth"
        )
        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold_idx} not found. Skipping.")
            continue

        logger.info(f"Inference with model fold {fold_idx}...")
        model = SiameseModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)
                meta_features = batch["meta_features"].to(device)

                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    meta_features,
                )
                probs = F.softmax(logits, dim=1)
                fold_probs.append(probs.cpu().numpy())

        ensemble_probs += np.concatenate(fold_probs, axis=0)
        models_found += 1

        # Cleanup
        del model
        torch.cuda.empty_cache()

    if models_found > 0:
        avg_probs = ensemble_probs / models_found
    else:
        logger.error("No models found for validation inference.")
        return

    # Calculate Metric
    # Targets: winner_model_a, winner_model_b, winner_tie
    y_true = val_processed[["winner_model_a", "winner_model_b", "winner_tie"]].values

    # Log Loss with eps=auto (default in sklearn is effectively auto/1e-15)
    metric = log_loss(y_true, avg_probs)

    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate per-sample Cross Entropy
    eps = 1e-15
    pred_clipped = np.clip(avg_probs, eps, 1 - eps)
    sample_losses = -np.sum(y_true * np.log(pred_clipped), axis=1)

    val_processed["error"] = sample_losses

    # Features to correlate
    # Note: load_data creates 'len_prompt', 'len_a', 'len_b' before normalization
    analysis_cols = ["len_prompt", "len_a", "len_b"]

    correlations = {}
    for col in analysis_cols:
        if col in val_processed.columns:
            corr = val_processed["error"].corr(val_processed[col])
            correlations[col] = corr

    # Calculate length difference correlation
    if "len_a" in val_processed.columns and "len_b" in val_processed.columns:
        val_processed["len_diff"] = (
            val_processed["len_a"] - val_processed["len_b"]
        ).abs()
        correlations["len_diff_abs"] = val_processed["error"].corr(
            val_processed["len_diff"]
        )

    print("Failure Analysis - Correlation with Error:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.4f}")

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = 1.0102717496437368

    if metric < THRESHOLD:
        logger.info(
            f"Metric {metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        logger.info(
            f"Metric {metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
