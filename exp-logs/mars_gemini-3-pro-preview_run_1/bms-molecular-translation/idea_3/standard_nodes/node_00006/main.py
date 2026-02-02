import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import random
import time

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import InChiDataset, get_transforms, collate_fn
from library.model import VisualTransformer
from library.tokenizer import Tokenizer
from library.train import train_one_epoch, seed_everything, greedy_decode
from library.utils import compute_levenshtein


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    Config.BATCH_SIZE = 256  # Increase batch size for A100
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 4

    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Enable cudnn benchmark for constant image size
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Data Loading ---")

    # Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA}"
        )
    if not os.path.exists(Config.VAL_METADATA):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Subsample for Fast Baseline Training
    # Using 30,000 training samples to get a model that learns basic syntax
    train_subset_size = 30000
    if len(df_train) > train_subset_size:
        print(
            f"Subsampling training data from {len(df_train)} to {train_subset_size}..."
        )
        df_train = df_train.sample(
            n=train_subset_size, random_state=Config.SEED
        ).reset_index(drop=True)

    # Subsample Validation for Speed
    # Using 5,000 samples for validation to ensure metric computation fits in time limit
    val_subset_size = 5000
    if len(df_val) > val_subset_size:
        print(f"Subsampling validation data from {len(df_val)} to {val_subset_size}...")
        df_val = df_val.sample(n=val_subset_size, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Initialize Tokenizer
    # We assume the tokenizer was built on the full training set during the metadata phase or we build it now.
    # We pass debug=False to ensure full vocab coverage if it needs to be built.
    tokenizer = Tokenizer(load_cached_data=True, debug=False)

    # Create Datasets
    train_dataset = InChiDataset(df_train, tokenizer, transform=get_transforms("train"))
    val_dataset = InChiDataset(df_val, tokenizer, transform=get_transforms("valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n--- Model Initialization ---")
    model = VisualTransformer(vocab_size=len(tokenizer))
    model = model.to(device)

    # Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch, scheduler
        )
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 5. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation ---")
    model.eval()

    val_predictions = []
    val_ground_truths = []
    val_image_ids = []

    # We use a slightly reduced max_len for inference speed, covering 99% of cases
    INFERENCE_MAX_LEN = 275

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            original_texts = batch["original_text"]

            # Use autocast for inference speed
            with autocast():
                generated_seqs = greedy_decode(
                    model, images, tokenizer, device, max_len=INFERENCE_MAX_LEN
                )

            pred_texts = [tokenizer.sequence_to_text(s) for s in generated_seqs]

            val_predictions.extend(pred_texts)
            val_ground_truths.extend(original_texts)
            val_image_ids.extend(batch["image_id"])

            if i % 10 == 0:
                print(f"Validation Step {i}/{len(val_loader)}")

    # Compute Metric
    final_metric = compute_levenshtein(val_predictions, val_ground_truths)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    import nltk

    # Calculate Levenshtein distance per sample
    errors = []
    lengths = []

    for p, t in zip(val_predictions, val_ground_truths):
        dist = nltk.edit_distance(p, t)
        errors.append(dist)
        lengths.append(len(t))

    analysis_df = pd.DataFrame({"error": errors, "target_length": lengths})

    correlation = analysis_df["error"].corr(analysis_df["target_length"])
    print(
        f"Correlation between Error Magnitude and Target Sequence Length: {correlation:.4f}"
    )

    print("Example Failures (Top 3 highest errors):")
    analysis_df["prediction"] = val_predictions
    analysis_df["target"] = val_ground_truths
    worst_cases = analysis_df.sort_values("error", ascending=False).head(3)
    for _, row in worst_cases.iterrows():
        print(f"Target: {row['target']}")
        print(f"Pred:   {row['prediction']}")
        print(f"Dist:   {row['error']}")
        print("-" * 20)

    # -------------------------------------------------------------------------
    # 7. Test Inference & Submission
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)
    print(f"Test dataset size: {len(df_test)}")

    test_dataset = InChiDataset(df_test, tokenizer, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_ids = []
    test_preds = []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            ids = batch["image_id"]

            with autocast():
                generated_seqs = greedy_decode(
                    model, images, tokenizer, device, max_len=INFERENCE_MAX_LEN
                )

            pred_texts = [tokenizer.sequence_to_text(s) for s in generated_seqs]

            test_ids.extend(ids)
            test_preds.extend(pred_texts)

            if step % 50 == 0:
                print(f"Test Inference Step {step}/{len(test_loader)}")

    # Save Submission
    sub_df = pd.DataFrame({"image_id": test_ids, "InChI": test_preds})
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
