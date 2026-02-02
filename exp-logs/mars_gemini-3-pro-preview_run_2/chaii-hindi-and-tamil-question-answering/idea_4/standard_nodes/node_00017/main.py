import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from transformers import logging as transformers_logging
from datasets import Dataset as HFDataset

from library.config import Config
from library.utils import (
    set_seed,
    postprocess_qa_predictions,
    compute_metrics,
    save_submission,
    jaccard,
)
from library.data import load_processed_data, prepare_train_features, QADataset
from library.model import get_model, get_tokenizer
from library.engine import train_fn, eval_fn

# Silence transformers warnings
transformers_logging.set_verbosity_error()


def run_training():
    """
    Trains the model using Group K-Fold Cross Validation.
    """
    print("Loading raw training data...")
    df_train_raw = pd.read_csv(Config.TRAIN_PATH)

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Group K-Fold
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    groups = df_train_raw["context"]

    for fold, (train_idx, _) in enumerate(gkf.split(df_train_raw, groups=groups)):
        print(f"\n=== Training Fold {fold + 1}/{Config.N_FOLDS} ===")

        # We use the training portion of the fold.
        # Note: In this strategy, we are training an ensemble on the full 'train.csv' data
        # split into folds. We are NOT validating on the fold's validation split because
        # we have a dedicated hold-out set (metadata/val.csv).
        # However, to create diverse models, we train on the train_idx of the fold.
        # Alternatively, one could train on the full dataset with different seeds,
        # but K-Fold is the requested strategy.

        df_fold_train = df_train_raw.iloc[train_idx].reset_index(drop=True)

        # Prepare features for this fold
        print(f"Processing features for fold {fold + 1}...")
        hf_dataset = HFDataset.from_pandas(df_fold_train)
        processed_dataset = hf_dataset.map(
            lambda x: prepare_train_features(x, tokenizer),
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc=f"Tokenizing Fold {fold + 1}",
        )

        # Create Dataset and DataLoader
        train_data = processed_dataset.to_pandas()
        train_dataset = QADataset(train_data, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model and Optimizer
        model = get_model(pretrained=True)
        model.to(Config.DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        num_train_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=num_warmup_steps
        )  # Simplified warmup

        # Training Loop
        for epoch in range(Config.EPOCHS):
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
            train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Cleanup
        del model, optimizer, scheduler, train_loader, train_dataset, train_data
        torch.cuda.empty_cache()


def run_validation():
    """
    Evaluates the ensemble on the hold-out validation set.
    """
    print("\n=== Running Validation on Hold-out Set ===")

    tokenizer = get_tokenizer()

    # Load raw validation data (for ground truth and text)
    df_val_raw = pd.read_csv(Config.VAL_PATH)
    hf_val_raw = HFDataset.from_pandas(df_val_raw)

    # Load processed validation features (using library function)
    df_val_features = load_processed_data(tokenizer, split="val", load_cached_data=True)

    val_dataset = QADataset(df_val_features, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize accumulators for ensemble logits
    # We need to know the size of the output.
    # eval_fn returns concatenated numpy arrays.
    # We run one pass to get shapes, or just accumulate in lists.

    ensemble_start_logits = None
    ensemble_end_logits = None

    for fold in range(Config.N_FOLDS):
        print(f"Inference with model fold {fold}...")
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

        model = get_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        start_logits, end_logits = eval_fn(val_loader, model, Config.DEVICE)

        if ensemble_start_logits is None:
            ensemble_start_logits = start_logits
            ensemble_end_logits = end_logits
        else:
            ensemble_start_logits += start_logits
            ensemble_end_logits += end_logits

        del model
        torch.cuda.empty_cache()

    # Average logits
    ensemble_start_logits /= Config.N_FOLDS
    ensemble_end_logits /= Config.N_FOLDS

    # Post-process
    # We need to pass the features as a list of dicts for the postprocess function
    features_list = df_val_features.to_dict("records")

    predictions = postprocess_qa_predictions(
        examples=hf_val_raw,
        features=features_list,
        predictions=(ensemble_start_logits, ensemble_end_logits),
    )

    # Compute Metric
    metric = compute_metrics(predictions, df_val_raw)
    print(f"Final Validation Metric: {metric}")

    # === Failure Analysis ===
    print("\n=== Failure Analysis ===")

    # Calculate Jaccard per sample
    scores = []
    context_lens = []
    question_lens = []

    for _, row in df_val_raw.iterrows():
        pid = str(row["id"])
        pred_text = predictions.get(pid, "")
        gt_text = row["answer_text"]

        score = jaccard(gt_text, pred_text)
        scores.append(score)

        context_lens.append(len(str(row["context"])))
        question_lens.append(len(str(row["question"])))

    errors = 1.0 - np.array(scores)

    # Correlations
    corr_ctx = np.corrcoef(errors, context_lens)[0, 1]
    corr_q = np.corrcoef(errors, question_lens)[0, 1]

    print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Length): {corr_q:.4f}")

    return metric


def run_submission():
    """
    Generates the submission file using the ensemble on the test set.
    """
    print("\n=== Generating Submission ===")

    tokenizer = get_tokenizer()

    # Load raw test data
    df_test_raw = pd.read_csv(Config.TEST_PATH)
    hf_test_raw = HFDataset.from_pandas(df_test_raw)

    # Load processed test features
    df_test_features = load_processed_data(
        tokenizer, split="test", load_cached_data=True
    )

    test_dataset = QADataset(df_test_features, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ensemble_start_logits = None
    ensemble_end_logits = None

    for fold in range(Config.N_FOLDS):
        print(f"Inference with model fold {fold}...")
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

        model = get_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        start_logits, end_logits = eval_fn(test_loader, model, Config.DEVICE)

        if ensemble_start_logits is None:
            ensemble_start_logits = start_logits
            ensemble_end_logits = end_logits
        else:
            ensemble_start_logits += start_logits
            ensemble_end_logits += end_logits

        del model
        torch.cuda.empty_cache()

    # Average logits
    ensemble_start_logits /= Config.N_FOLDS
    ensemble_end_logits /= Config.N_FOLDS

    # Post-process
    features_list = df_test_features.to_dict("records")

    predictions = postprocess_qa_predictions(
        examples=hf_test_raw,
        features=features_list,
        predictions=(ensemble_start_logits, ensemble_end_logits),
    )

    # Save
    submission_path = "./submission/submission.csv"
    save_submission(predictions, submission_path)
    print(f"Submission saved to {submission_path}")


def main():
    set_seed(Config.SEED)

    # 1. Train
    run_training()

    # 2. Validate
    metric = run_validation()

    # 3. Submit (Conditional)
    # Threshold from instructions: 0.2522202380952381
    THRESHOLD = 0.2522202380952381

    if metric > THRESHOLD:
        run_submission()
    else:
        print(
            f"Validation metric {metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
