import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import set_seed, compute_average_jaccard, cleanup, jaccard
from library.data_loader import (
    get_tokenizer,
    prepare_train_features,
    prepare_test_features,
    QADataset,
)
from library.model_arch import get_model
from library.train_runner import run_training
from library.inference_engine import ensemble_and_postprocess, generate_submission


def main():
    # 1. Setup & Reproducibility
    set_seed(Config.SEED)
    print(f"Starting execution. Device: {Config.DEVICE}")

    # 2. Load Training Data
    print(f"Loading training data from {Config.TRAIN_CSV}...")
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Ensure text columns are strings
    for col in ["context", "question", "answer_text"]:
        train_df[col] = train_df[col].astype(str)

    tokenizer = get_tokenizer()

    # 3. 5-Fold Cross-Validation Training
    print(f"Initializing {Config.N_FOLDS}-Fold Cross-Validation...")
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    folds = list(gkf.split(train_df, groups=train_df["context"]))

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*20} Processing Fold {fold + 1}/{Config.N_FOLDS} {'='*20}")

        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Prepare Features
        # We use prepare_train_features for both train and val splits within the fold
        # because we need labels (start/end positions) to compute Loss for Early Stopping.
        print("Preparing fold training features...")
        train_features = prepare_train_features(fold_train_df, tokenizer)

        print("Preparing fold validation features (for loss monitoring)...")
        val_features = prepare_train_features(fold_val_df, tokenizer)

        # Create Datasets
        train_dataset = QADataset(train_features, mode="train")
        val_dataset = QADataset(val_features, mode="train")

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Scheduler
        model = get_model()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Run Training
        save_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_best_model.pth")

        print(f"Training fold {fold + 1}...")
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.DEVICE,
            epochs=Config.EPOCHS,
            save_path=save_path,
            patience=Config.PATIENCE,
        )

        # Cleanup to free memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        cleanup()

    # 4. Hold-out Validation Evaluation
    print(f"\n{'='*20} Hold-out Validation Evaluation {'='*20}")
    print(f"Loading hold-out validation data from {Config.VAL_CSV}...")
    val_holdout_df = pd.read_csv(Config.VAL_CSV)

    for col in ["context", "question", "answer_text"]:
        val_holdout_df[col] = val_holdout_df[col].astype(str)

    # Prepare features for inference (no sampling, full sliding window)
    print("Preparing hold-out features for inference...")
    val_holdout_features = prepare_test_features(val_holdout_df, tokenizer)

    # Run Ensemble Inference using all trained folds
    fold_indices = list(range(Config.N_FOLDS))
    print("Running ensemble inference on hold-out set...")
    preds_df = ensemble_and_postprocess(
        val_holdout_df, val_holdout_features, fold_indices
    )

    # Merge predictions with ground truth
    merged_df = val_holdout_df.merge(preds_df, on="id", how="left")
    merged_df["PredictionString"] = merged_df["PredictionString"].fillna('"dummy text"')

    # Helper to strip quotes from prediction for metric calculation
    # (PredictionString is formatted as "answer", but GT is answer)
    def clean_prediction(text):
        text = str(text).strip()
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            return text[1:-1]
        return text

    ground_truths = merged_df["answer_text"].tolist()
    predictions = merged_df["PredictionString"].apply(clean_prediction).tolist()

    # Compute Metric
    final_metric = compute_average_jaccard(ground_truths, predictions)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")

    # Calculate per-sample error
    merged_df["jaccard"] = [
        jaccard(gt, dt) for gt, dt in zip(ground_truths, predictions)
    ]
    merged_df["error"] = 1.0 - merged_df["jaccard"]

    # Calculate meta-features
    merged_df["context_len"] = merged_df["context"].apply(len)
    merged_df["question_len"] = merged_df["question"].apply(len)

    # Compute correlations
    corr_ctx = merged_df["error"].corr(merged_df["context_len"])
    corr_que = merged_df["error"].corr(merged_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx}")
    print(f"Correlation (Error vs Question Length): {corr_que}")

    # 6. Submission Generation
    THRESHOLD = 0.4804191919191919

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Proceeding to submission generation."
        )
        # We disable cache loading for the test set to ensure a clean run
        generate_submission(load_cached_data=False)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
