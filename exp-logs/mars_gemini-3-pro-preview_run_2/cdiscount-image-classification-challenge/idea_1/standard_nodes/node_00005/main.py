import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time

# Import provided library modules
from library.config import (
    WORKING_DIR,
    SUBMISSION_FILE,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    EPOCHS,
    VAL_META,
    seed_everything,
)
from library.utils import load_checkpoint
from library.dataset import (
    CdiscountDataset,
    get_transforms,
    train_collate_fn,
    eval_collate_fn,
)
from library.model import get_model
from library.engine import train_model, validate, make_predictions


def perform_failure_analysis(model, val_loader, val_meta_path):
    """
    Analyzes model performance on the validation set to find correlations
    between errors and input features.
    """
    print("\n==== Performing Failure Analysis ====")
    model.eval()

    results = []

    # Run inference to get predictions and metadata
    with torch.no_grad():
        for i, (flattened_images, targets, product_ids, num_imgs) in enumerate(
            val_loader
        ):
            flattened_images = flattened_images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                output = model(flattened_images)
                probs = torch.softmax(output, dim=1)

            # Split back to products
            split_probs = torch.split(probs, num_imgs.tolist())

            for j, prod_probs in enumerate(split_probs):
                # Late fusion
                avg_prob = torch.mean(prod_probs, dim=0)
                pred_cat = torch.argmax(avg_prob).item()
                true_cat = targets[j].item()
                p_id = product_ids[j].item()
                n_imgs = num_imgs[j].item()

                is_error = 1 if pred_cat != true_cat else 0

                results.append(
                    {"product_id": p_id, "num_images": n_imgs, "is_error": is_error}
                )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Load metadata to get BSON length (proxy for data size/quality)
    df_meta = pd.read_csv(val_meta_path)

    # Merge results with metadata
    df_analysis = df_results.merge(
        df_meta[["product_id", "bson_length"]], on="product_id", how="left"
    )

    # Calculate correlations
    # We look at correlation between 'is_error' and features
    corr_imgs = df_analysis["is_error"].corr(df_analysis["num_images"])
    corr_size = df_analysis["is_error"].corr(df_analysis["bson_length"])

    print(f"Analysis based on {len(df_analysis)} validation samples.")
    print(f"Correlation between Error and Number of Images: {corr_imgs:.4f}")
    print(f"Correlation between Error and BSON Record Size: {corr_size:.4f}")

    # Grouped analysis
    print("\nError Rate by Number of Images:")
    print(df_analysis.groupby("num_images")["is_error"].mean())


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    # Limit training size for fast baseline execution (e.g., 500k samples)
    # Use FULL validation and test sets as required.
    # Increasing to 2M samples to improve performance within time limits.
    TRAIN_LIMIT = 2000000

    print("Initializing Datasets...")
    train_dataset = CdiscountDataset(
        mode="train", transform=get_transforms("train"), debug_size=TRAIN_LIMIT
    )
    val_dataset = CdiscountDataset(
        mode="val",
        transform=get_transforms("val"),
        debug_size=None,  # Full validation set
    )
    test_dataset = CdiscountDataset(
        mode="test", transform=get_transforms("test"), debug_size=None  # Full test set
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )

    # 3. Model
    print("Initializing Model...")
    model = get_model(pretrained=True)
    model = model.to(DEVICE)

    # 4. Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler: OneCycleLR for super-convergence in few epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    print("Starting Training...")
    start_time = time.time()

    # Train for defined EPOCHS (default 2 in config)
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        device=DEVICE,
        patience=2,
    )

    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # 5. Validation & Metric
    print("Loading best checkpoint for final validation...")
    _, best_acc = load_checkpoint(model)

    print("Running full validation...")
    val_acc, val_loss = validate(val_loader, model, criterion, device=DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, VAL_META)

    # 7. Submission
    print("Generating submission file...")
    make_predictions(test_loader, model, device=DEVICE, output_file=SUBMISSION_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
