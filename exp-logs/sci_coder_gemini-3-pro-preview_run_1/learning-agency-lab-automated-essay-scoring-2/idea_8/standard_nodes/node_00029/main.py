import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.configuration import Config, seed_everything
from library.utilities import get_logger, compute_metrics
from library.training import run_training
from library.meta_modeling import run_stacking
from library.dataset import load_supervised_data, get_tokenizer, Collate, EssayDataset
from library.modeling import EssayModel
from library.feature_engineering import FeatureEngineer

# Initialize Logger
logger = get_logger("RunFile")


def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # === 1. Configuration Adjustments for Fast Baseline ===
    # Adjust epochs to ensure execution within the time limit
    logger.info("Adjusting configuration for fast baseline execution...")
    Config.EPOCHS = 1
    # We skip explicit MLM pre-training to prioritize supervised fine-tuning time
    # The training script handles missing MLM checkpoints by falling back to the base HF model

    # === 2. Level 1 Training ===
    logger.info("Starting Level 1 Training (DeBERTa-v3-Large)...")
    # This runs 5-fold CV on metadata/train.csv and saves checkpoints + OOF
    run_training(debug=False, load_cached_data=True)

    # === 3. Evaluation on Hold-out Validation Set ===
    logger.info("Starting Evaluation on Hold-out Validation Set (metadata/val.csv)...")

    # Load validation data
    tokenizer = get_tokenizer()
    val_dataset = load_supervised_data("val", tokenizer, load_cached_data=True)

    # Setup DataLoader
    collate_fn = Collate(tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Inference Loop (Ensemble of 5 Folds)
    fold_preds = []
    device = Config.DEVICE

    for fold in range(Config.NUM_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(ckpt_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {ckpt_path}. Skipping."
            )
            continue

        logger.info(f"Running inference with Fold {fold+1} model...")

        # Load Model
        model = EssayModel(pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Use mixed precision for faster inference
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)

                preds.append(logits.view(-1).cpu().numpy())

        fold_preds.append(np.concatenate(preds))

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if not fold_preds:
        logger.error("No predictions generated. Exiting.")
        return

    # Average predictions across folds
    avg_preds = np.mean(fold_preds, axis=0)

    # Get True Scores
    true_scores = val_dataset.df["score"].values

    # Compute Metric
    metrics = compute_metrics(true_scores, avg_preds)
    final_qwk = metrics["qwk"]

    # Print Required Output
    print(f"Final Validation Metric: {final_qwk}")

    # === 4. Failure Analysis ===
    logger.info("Performing Failure Analysis...")

    # Calculate absolute residuals (error magnitude)
    residuals = np.abs(true_scores - avg_preds)

    # Extract meta-features for correlation analysis
    fe = FeatureEngineer()
    val_features_df = fe.extract_features(val_dataset.df)

    analysis_cols = [
        "char_count",
        "word_count",
        "sentence_count",
        "unique_word_count",
        "avg_word_len",
    ]

    print("Correlation between Error Magnitude and Input Features:")
    for col in analysis_cols:
        if col in val_features_df.columns:
            feat_values = val_features_df[col].values
            # Handle potential NaNs or constant values
            if np.std(feat_values) > 0 and np.std(residuals) > 0:
                corr = np.corrcoef(residuals, feat_values)[0, 1]
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: N/A (Constant or NaN)")

    # === 5. Submission Generation ===
    THRESHOLD = 0.8274925140324321

    if final_qwk > THRESHOLD:
        logger.info(
            f"Validation metric ({final_qwk:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # run_stacking handles:
        # 1. Loading OOF (Train) and Test Level 1 predictions
        # 2. Extracting meta-features
        # 3. Training the Meta-Learner (Ridge)
        # 4. Generating final submission.csv
        run_stacking(debug=False, load_cached_data=True)
    else:
        logger.info(
            f"Validation metric ({final_qwk:.6f}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
