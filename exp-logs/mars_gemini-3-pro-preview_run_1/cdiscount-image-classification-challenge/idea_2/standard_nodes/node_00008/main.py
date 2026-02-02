import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.dataset import BSONProductDataset, collate_fn
from library.model import HierarchicalResNet50
from library.train import Trainer
from library.evaluate import Evaluator


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    print("Initializing Datasets...")

    # Train Dataset - Subsampled for Fast Baseline
    # We load the dataset and then subsample the internal DataFrame to ensure
    # the training loop finishes quickly within the allocated time.
    train_dataset = BSONProductDataset(mode="train")

    # Target sample size for fast baseline training
    TRAIN_SAMPLE_SIZE = 200000

    if len(train_dataset.df) > TRAIN_SAMPLE_SIZE:
        print(
            f"Subsampling training set from {len(train_dataset.df)} to {TRAIN_SAMPLE_SIZE} records..."
        )
        train_dataset.df = train_dataset.df.sample(
            n=TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        print(f"Using full training set ({len(train_dataset.df)} records)...")

    # Validation Dataset - Full set required for metric calculation
    val_dataset = BSONProductDataset(mode="val")
    print(f"Using full validation set ({len(val_dataset.df)} records)...")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = HierarchicalResNet50()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler setup
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 4. Training Loop
    trainer = Trainer(model, train_loader, val_loader, device)

    best_val_acc = 0.0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss, train_acc = trainer.train_epoch(optimizer, scheduler, epoch + 1)

        # Validate
        val_loss, val_acc_l3 = trainer.validate()

        # Save Best Model
        if val_acc_l3 > best_val_acc:
            best_val_acc = val_acc_l3
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    print(f"Training finished. Best L3 Accuracy: {best_val_acc}")

    # 5. Final Validation Metric
    # Load best model weights
    if os.path.exists(Config.MODEL_CHECKPOINT):
        print(f"Loading best model from {Config.MODEL_CHECKPOINT}...")
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    evaluator = Evaluator(model, device)
    print("Computing Final Validation Metric on full validation set...")
    final_metric = evaluator.validate(val_loader)

    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    # Create a lookup for bson_length from the validation dataframe
    id_to_len = dict(zip(val_dataset.df["sample_id"], val_dataset.df["bson_length"]))

    results_analysis = []

    # Collect error statistics
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            targets = batch["labels"]["target"].to(device)
            ids = batch["ids"].numpy()
            masks = batch["mask"]  # CPU tensor

            outputs = model(images)
            _, preds = torch.max(outputs["target"], 1)

            # Identify errors (0 = correct, 1 = error)
            is_error = (preds != targets).cpu().numpy().astype(int)
            # Count valid images per product
            num_images = masks.sum(dim=1).numpy()

            for i in range(len(ids)):
                results_analysis.append(
                    {
                        "id": ids[i],
                        "is_error": is_error[i],
                        "num_images": num_images[i],
                        "bson_length": id_to_len.get(ids[i], 0),
                    }
                )

    df_analysis = pd.DataFrame(results_analysis)

    # Calculate Correlations
    if not df_analysis.empty:
        corr_imgs = df_analysis["is_error"].corr(df_analysis["num_images"])
        corr_len = df_analysis["is_error"].corr(df_analysis["bson_length"])

        print(f"Correlation between Error and Num_Images: {corr_imgs}")
        print(f"Correlation between Error and Bson_Length: {corr_len}")

        if abs(corr_imgs) > 0.05:
            direction = "positive" if corr_imgs > 0 else "negative"
            print(
                f"Insight: There is a {direction} correlation between number of images and error rate."
            )

    # 7. Submission
    THRESHOLD = 0.6306776302037904

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = BSONProductDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        evaluator.generate_submission(test_loader, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
