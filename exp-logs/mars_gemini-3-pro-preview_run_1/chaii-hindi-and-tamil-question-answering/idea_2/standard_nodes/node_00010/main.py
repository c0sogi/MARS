import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.model_handler import get_model, get_tokenizer, save_model
from library.data_loader import get_processed_data, QADataset
from library.post_processor import save_submission
from library.trainer import train_fn, eval_fn, predict_fn


def main():
    # 1. Setup
    # -------------------------------------------------------------------------
    set_seed(Config.seed)
    device = Config.device

    # Ensure we use the full dataset for best performance.
    # The dataset size (802 train samples) is small enough for fast training within the time limit.
    Config.debug = False

    print(f"Device: {device}")

    # 2. Data Preparation
    # -------------------------------------------------------------------------
    tokenizer = get_tokenizer()

    # Load processed features (sliding windows)
    # load_cached_data=True allows using pre-computed parquet files if available
    train_features, val_features, test_features = get_processed_data(
        tokenizer, load_cached_data=True
    )

    # Load raw examples for validation and testing (needed for metrics/inference)
    raw_val = pd.read_csv(Config.val_path)
    raw_test = pd.read_csv(Config.test_path)

    # Create Datasets
    train_dataset = QADataset(train_features, mode="train")
    val_dataset = QADataset(val_features, mode="val")
    test_dataset = QADataset(test_features, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    model = get_model()  # Loads pre-trained XLM-Roberta

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_score = -1.0

    print("Starting training...")
    for epoch in range(Config.epochs):
        # Train one epoch
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Evaluate on validation set
        val_score = eval_fn(model, val_loader, raw_val, val_features, device)

        # Save best model
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            save_model(model, tokenizer, Config.model_output_dir)
        else:
            print(f"Score {val_score} did not improve best {best_score}.")

    # 5. Final Validation and Failure Analysis
    # -------------------------------------------------------------------------
    # Print the required metric format
    print(f"Final Validation Metric: {best_score}")

    # Load the best model for analysis and inference
    print("Loading best model for analysis...")
    best_model = get_model(weights_path=Config.model_output_dir)

    # Generate predictions on validation set for failure analysis
    print("Running inference on validation set for failure analysis...")
    val_predictions = predict_fn(best_model, val_loader, raw_val, val_features, device)

    # Calculate error per sample and correlate with features
    analysis_data = []

    for idx, row in raw_val.iterrows():
        ex_id = row["id"]
        ground_truth = str(row["answer_text"])
        prediction = val_predictions.get(ex_id, "")

        # Calculate Jaccard score
        score = jaccard(ground_truth, prediction)
        error = 1.0 - score

        # Extract features
        context_len = len(str(row["context"]))
        question_len = len(str(row["question"]))

        analysis_data.append(
            {"error": error, "context_len": context_len, "question_len": question_len}
        )

    df_analysis = pd.DataFrame(analysis_data)

    # Calculate correlations
    if not df_analysis.empty:
        correlations = df_analysis.corr()["error"].drop("error")
        print("\n==== Failure Analysis: Correlation with Error ====")
        print(correlations)
    else:
        print("Analysis dataframe is empty.")

    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.3617383311133311

    if best_score > THRESHOLD:
        print(
            f"\nValidation score ({best_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        test_predictions = predict_fn(
            best_model, test_loader, raw_test, test_features, device
        )

        # Save submission
        save_submission(test_predictions, Config.submission_path)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nValidation score ({best_score}) is below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
