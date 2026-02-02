import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import nltk
from scipy.stats import pearsonr

# Add library to path
sys.path.append("./library")

from library.config import Config
from library.utils import seed_everything, compute_levenshtein, save_checkpoint
from library.tokenizer import InChITokenizer
from library.dataset import load_dataframe, ChemicalDataset, get_transforms
from library.model import ViT2InChI
from library.trainer import train_one_epoch, validate, predict_and_submit


def run():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Modify Config for Fast Baseline and Hardware Optimization
    Config.EPOCHS = 1  # Limit epochs for speed
    Config.BATCH_SIZE = 256  # Increase batch size for A100 GPU
    Config.NUM_WORKERS = 12  # Utilize all available vCPUs

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    train_df = load_dataframe("train")
    val_df = load_dataframe("val")

    # Subset training data for a fast baseline (50k samples)
    if len(train_df) > 50000:
        train_df = train_df.sample(n=50000, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Subset validation data for monitoring during training (5k samples)
    val_subset_df = val_df
    if len(val_subset_df) > 5000:
        val_subset_df = val_subset_df.sample(
            n=5000, random_state=Config.SEED
        ).reset_index(drop=True)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation subset size (for monitoring): {len(val_subset_df)}")

    tokenizer = InChITokenizer()
    transforms = get_transforms(Config.IMAGE_SIZE)

    # Create Datasets
    train_dataset = ChemicalDataset(
        train_df, tokenizer, transform=transforms, mode="train"
    )
    val_subset_dataset = ChemicalDataset(
        val_subset_df, tokenizer, transform=transforms, mode="val"
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_subset_loader = torch.utils.data.DataLoader(
        val_subset_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing ViT2InChI model...")
    model = ViT2InChI().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    # Scheduler: Cosine Annealing for 1 epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training...")
    best_score = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate on subset
        val_loss, val_score = validate(
            model, val_subset_loader, criterion, device, tokenizer
        )

        scheduler.step()

        # Save checkpoint
        if val_score < best_score:
            best_score = val_score
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, Config.BEST_MODEL_PATH
            )
            print(
                f"Epoch {epoch}: New best model saved with Levenshtein distance {best_score:.4f}"
            )

    # -------------------------------------------------------------------------
    # 5. Full Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("FULL VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load best model
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create DataLoader for the ENTIRE validation set
    val_full_dataset = ChemicalDataset(
        val_df, tokenizer, transform=transforms, mode="val"
    )
    val_full_loader = torch.utils.data.DataLoader(
        val_full_dataset,
        batch_size=Config.BATCH_SIZE,  # Use large batch size for inference speed
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    print(f"Running inference on full validation set ({len(val_df)} samples)...")
    with torch.no_grad():
        for images, targets in val_full_loader:
            images = images.to(device)

            # Greedy decoding
            pred_indices = model.predict(
                images, max_len=Config.MAX_TEXT_LEN, device=device
            )

            # Convert to text
            pred_texts = tokenizer.batch_decode(pred_indices)
            target_texts = tokenizer.batch_decode(targets)

            all_preds.extend(pred_texts)
            all_targets.extend(target_texts)

    # Compute Final Metric
    final_metric = compute_levenshtein(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Sequence Length and Error
    print("Analyzing failure modes...")
    errors = []
    lengths = []

    for pred, target in zip(all_preds, all_targets):
        dist = nltk.edit_distance(pred, target)
        errors.append(dist)
        lengths.append(len(target))

    if len(errors) > 1:
        corr, _ = pearsonr(lengths, errors)
        print(
            f"Correlation between Target Sequence Length and Levenshtein Error: {corr:.4f}"
        )
        if corr > 0.3:
            print(
                "-> Analysis: Strong positive correlation. The model struggles with longer/more complex molecules."
            )
        else:
            print(
                "-> Analysis: Weak correlation. Errors are distributed across sequence lengths."
            )

    # -------------------------------------------------------------------------
    # 6. Test Submission
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("GENERATING SUBMISSION")
    print("=" * 40)

    test_df = load_dataframe("test")
    test_dataset = ChemicalDataset(
        test_df, tokenizer, transform=transforms, mode="test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    output_csv = "./submission/submission.csv"
    predict_and_submit(model, test_loader, device, tokenizer, save_path=output_csv)


if __name__ == "__main__":
    run()
