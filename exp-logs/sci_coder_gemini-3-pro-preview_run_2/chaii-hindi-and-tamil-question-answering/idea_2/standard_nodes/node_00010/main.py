import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from transformers import get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_average_jaccard, jaccard
from library.model_factory import get_model, get_tokenizer
from library.data_factory import (
    get_train_dataset,
    get_val_dataset,
    get_test_dataset,
    QADataset,
)
from library.trainer import TrainRunner
from library.predictor import InferenceEngine, generate_submission


def main():
    # 1. Setup & Configuration
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set reproducibility
    set_seed(Config.seed)

    # Override Config for Fast Baseline
    # We reduce epochs and folds to ensure the script completes quickly within the limit.
    Config.epochs = 3
    Config.n_folds = 3

    print(f"Starting Optimized Run (DistilBERT, 3 Epochs)...")
    print(f"Device: {Config.device}")
    print(f"Config: {Config.n_folds} Folds, {Config.epochs} Epochs")

    # 2. Data Loading
    print("\n[Data Loading]")
    tokenizer = get_tokenizer()

    # Load Training Data
    # We need both features (for training) and raw DF (for GroupKFold splitting)
    print("Loading training data...")
    train_dataset_full = get_train_dataset(tokenizer, load_cached_data=True)
    train_features_df = train_dataset_full.features
    raw_train_df = pd.read_csv(Config.train_meta_path)

    # Load Validation Data (Hold-out)
    print("Loading validation data...")
    val_dataset, raw_val_df, val_features_df = get_val_dataset(
        tokenizer, load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Training Loop (Group K-Fold)
    print(f"\n[Training] Starting Group K-Fold ({Config.n_folds} folds)...")

    gkf = GroupKFold(n_splits=Config.n_folds)
    # Split based on context to prevent leakage
    splits = list(gkf.split(raw_train_df, groups=raw_train_df["context"]))

    for fold_idx, (train_idx, _) in enumerate(splits):
        print(f"\n>>> Fold {fold_idx + 1}/{Config.n_folds}")

        # Map raw dataframe indices to feature indices
        # 1. Get IDs of the training samples for this fold
        fold_train_ids = set(raw_train_df.iloc[train_idx]["id"])

        # 2. Filter the features DataFrame to include only these IDs
        fold_train_features = train_features_df[
            train_features_df["example_id"].isin(fold_train_ids)
        ].reset_index(drop=True)

        # 3. Create Dataset and DataLoader
        fold_train_dataset = QADataset(fold_train_features)
        train_loader = DataLoader(
            fold_train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize Model & Optimizer
        model = get_model()
        model.to(Config.device)

        optimizer = AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        num_training_steps = len(train_loader) * Config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        # Initialize Trainer
        runner = TrainRunner(model, tokenizer, optimizer, scheduler, Config.device)

        # Run Training
        # We use the hold-out val_loader for monitoring/saving best model
        runner.run(train_loader, val_loader, val_features_df, raw_val_df, fold_idx)

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, runner, train_loader, fold_train_dataset
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    print("\n[Validation] Running Ensemble Inference...")

    inference_engine = InferenceEngine(device=Config.device)
    # Explicitly pass the current n_folds config
    inference_engine.load_ensemble(num_folds=Config.n_folds)

    # Predict
    val_preds_map = inference_engine.predict(val_loader, val_features_df, raw_val_df)

    # Compute Metric
    ground_truths = raw_val_df["answer_text"].tolist()
    # Ensure order matches
    predictions = [val_preds_map.get(row["id"], "") for _, row in raw_val_df.iterrows()]

    final_metric = compute_average_jaccard(ground_truths, predictions)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n[Failure Analysis] Analyzing Error Patterns...")

    scores = []
    context_lens = []
    question_lens = []

    for _, row in raw_val_df.iterrows():
        gt = row["answer_text"]
        pred = val_preds_map.get(row["id"], "")
        score = jaccard(gt, pred)
        scores.append(score)

        # Measure lengths
        context_lens.append(len(str(row["context"])))
        question_lens.append(len(str(row["question"])))

    scores = np.array(scores)
    context_lens = np.array(context_lens)
    question_lens = np.array(question_lens)

    # Error is inverse of score
    error_magnitude = 1.0 - scores

    # Calculate correlations
    if len(error_magnitude) > 1:
        # Check for constant input to avoid NaN in correlation
        if np.std(error_magnitude) > 0 and np.std(context_lens) > 0:
            corr_ctx = np.corrcoef(error_magnitude, context_lens)[0, 1]
            print(f"Correlation (Error vs Context Length): {corr_ctx:.10f}")
        else:
            print("Correlation (Error vs Context Length): Undefined (Zero Variance)")

        if np.std(error_magnitude) > 0 and np.std(question_lens) > 0:
            corr_q = np.corrcoef(error_magnitude, question_lens)[0, 1]
            print(f"Correlation (Error vs Question Length): {corr_q:.10f}")
        else:
            print("Correlation (Error vs Question Length): Undefined (Zero Variance)")
    else:
        print("Insufficient data for failure analysis.")

    # 6. Submission Generation
    THRESHOLD = 0.2522202380952381

    if final_metric > THRESHOLD:
        print(
            f"\n[Submission] Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset, raw_test_df, test_features_df = get_test_dataset(
            tokenizer, load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Predict
        test_preds_map = inference_engine.predict(
            test_loader, test_features_df, raw_test_df
        )

        # Save
        output_path = "./submission/submission.csv"
        generate_submission(test_preds_map, output_path=output_path)
        print(f"Submission saved to {output_path}")
    else:
        print(
            f"\n[Submission] Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
