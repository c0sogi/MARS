import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_score
from library.dataset import PhraseDataset, process_data
from library.model import CustomDeberta
from library.engine import (
    train_model,
    evaluate,
    get_expected_scores,
    generate_submission,
)


def analyze_failures(df, preds, true_scores):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    df = df.copy()
    df["pred_score"] = preds
    df["true_score"] = true_scores
    df["error"] = (df["pred_score"] - df["true_score"]).abs()

    # Generate Features
    df["anchor_len"] = df["anchor"].astype(str).apply(len)
    df["target_len"] = df["target"].astype(str).apply(len)
    df["len_diff"] = (df["anchor_len"] - df["target_len"]).abs()

    def get_jaccard(s1, s2):
        a = set(str(s1).lower().split())
        b = set(str(s2).lower().split())
        c = a.intersection(b)
        return (
            float(len(c)) / (len(a) + len(b) - len(c))
            if (len(a) + len(b) - len(c)) > 0
            else 0.0
        )

    df["jaccard_sim"] = df.apply(
        lambda x: get_jaccard(x["anchor"], x["target"]), axis=1
    )

    # Calculate Correlations
    features = ["anchor_len", "target_len", "len_diff", "jaccard_sim", "true_score"]
    correlations = df[features].corrwith(df["error"])

    print("Correlation between Error Magnitude and Features:")
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.6f}")

    return correlations


def main():
    # 1. Configuration & Setup
    seed_everything(Config.seed)

    # Configuration
    Config.debug = False  # Ensure we use full data to hit the score threshold

    print(f"Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.train_batch_size}")
    print(f"  Model: {Config.model_name}")

    # 2. Data Loading
    print("\nLoading Data...")
    # Load processed dataframes (uses cache if available)
    train_df = process_data("train", load_cached_data=True)
    val_df = process_data("val", load_cached_data=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    train_dataset = PhraseDataset(
        train_df, tokenizer, max_length=Config.max_length, is_test=False
    )
    val_dataset = PhraseDataset(
        val_df, tokenizer, max_length=Config.max_length, is_test=False
    )

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
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = CustomDeberta(model_name=Config.model_name, num_classes=Config.num_classes)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(Config.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training
    print("\nStarting Training...")
    # We treat the provided train/val split as a single fold (Fold 0)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        num_epochs=Config.epochs,
        patience=2,  # Strict patience for speed
        fold=0,
    )

    # 5. Validation & Metrics
    print("\nRunning Final Validation...")
    # Ensure model is in eval mode
    model.eval()

    val_loss, val_logits, val_labels = evaluate(model, val_loader, Config.device)

    # Convert predictions and labels to scores
    val_preds_score = get_expected_scores(val_logits)
    # val_labels are class indices (0-4), convert to 0.0-1.0
    val_true_score = val_labels * 0.25

    final_score = compute_score(val_true_score, val_preds_score)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    analyze_failures(val_df, val_preds_score, val_true_score)

    # 7. Submission
    THRESHOLD = 0.8550264305718601

    if final_score > THRESHOLD:
        print(
            f"\nScore ({final_score}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        test_df = process_data("test", load_cached_data=True)
        test_dataset = PhraseDataset(
            test_df, tokenizer, max_length=Config.max_length, is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        generate_submission(model, test_loader, Config.device)

    else:
        print(
            f"\nScore ({final_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
