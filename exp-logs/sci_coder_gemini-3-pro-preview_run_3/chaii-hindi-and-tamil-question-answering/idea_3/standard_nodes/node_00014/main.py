import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, jaccard
from library.data_manager import get_fold_dataloaders, get_cached_features, QADataset
from library.model_factory import get_tokenizer, get_qa_model
from library.trainer import run_tapt, train_qa_fold
from library.inference import predict_fold, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Adjust Config for Fast Baseline execution
    Config.EPOCHS = 3  # Reduced to ensure quick execution
    Config.TAPT_EPOCHS = 3  # Reduced to ensure quick execution
    Config.N_FOLDS = 1  # Use 1 fold (Train/Val split) to strictly use hold-out val

    # Initialize directories and seeds
    Config.setup()
    set_seed(Config.SEED)

    print("Configuration set up.")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Task-Adaptive Pretraining (TAPT)
    # -------------------------------------------------------------------------
    # Trains MLM on the domain text to adapt weights
    run_tapt()

    # -------------------------------------------------------------------------
    # 3. QA Model Training
    # -------------------------------------------------------------------------
    tokenizer = get_tokenizer()

    # Get DataLoaders
    # With N_FOLDS=1, this returns a single tuple (train_loader, val_loader)
    # corresponding to the original train.csv and val.csv split.
    fold_generator = get_fold_dataloaders(
        tokenizer, k_folds=Config.N_FOLDS, load_cached_data=True
    )

    for fold_idx, (train_loader, val_loader) in enumerate(fold_generator):
        print(f"\n=== Training QA Model Fold {fold_idx} ===")

        # Initialize model using the TAPT-adapted weights
        # TAPT_MODEL_PATH is a directory containing the saved model
        model = get_qa_model(model_path=Config.TAPT_MODEL_PATH)

        # Train the model
        train_qa_fold(model, train_loader, val_loader, fold_idx)

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n=== Performing Validation Assessment ===")

    # Load the hold-out validation metadata
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Prepare features for inference (is_training=False ensures no labels, just IDs/Offsets)
    # We use a distinct cache name to avoid conflicts
    val_features = get_cached_features(
        df_val,
        tokenizer,
        cache_name="val_holdout_features",
        load_cached_data=True,
        is_training=False,
    )

    val_dataset = QADataset(val_features)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Predict using the trained model (Fold 0)
    # predict_fold returns a dict: {example_id: prediction_string}
    val_predictions = predict_fold(fold_idx=0, dataloader=val_loader)

    # Calculate Jaccard Score
    scores = []
    error_magnitudes = []
    context_lengths = []
    question_lengths = []

    # Map predictions to ground truth
    # df_val contains: id, context, question, answer_text
    print("Calculating metrics...")
    for _, row in df_val.iterrows():
        eid = row["id"]
        gt_text = row["answer_text"]
        pred_text = val_predictions.get(
            eid, ""
        )  # Default to empty string if no prediction

        # Metric
        score = jaccard(gt_text, pred_text)
        scores.append(score)

        # For Failure Analysis
        error_magnitudes.append(1.0 - score)

        # Features for correlation
        # Simple word count
        ctx_len = len(str(row["context"]).split())
        q_len = len(str(row["question"]).split())

        context_lengths.append(ctx_len)
        question_lengths.append(q_len)

    final_metric = np.mean(scores)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    if len(error_magnitudes) > 1:
        # Calculate correlations
        # np.corrcoef returns matrix, take [0, 1]
        corr_ctx = np.corrcoef(error_magnitudes, context_lengths)[0, 1]
        corr_q = np.corrcoef(error_magnitudes, question_lengths)[0, 1]

        print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
        print(f"Correlation (Error vs Question Length): {corr_q:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold from requirements
    SUBMISSION_THRESHOLD = 0.3011529653320698

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
