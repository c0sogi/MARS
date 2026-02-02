import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    mean_average_precision,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import HotelDataset, get_transforms, get_label_encoder
from library.model import HotelModel
from library.loss import ArcFaceLoss
from library.trainer import train_fn, generate_submission


def pearson_corr(x, y):
    """Calculates Pearson correlation coefficient using numpy."""
    if len(x) != len(y):
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    seed_everything(Config.seed)

    # Override Config for Fast Baseline requirements
    Config.epochs = 4
    Config.batch_size = 128  # Optimized for A100

    print("=== Configuration ===")
    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Input Resolution: {Config.image_size}x{Config.image_size}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n=== Initializing Data ===")
    # Load/Fit Label Encoder
    label_encoder = get_label_encoder(
        Config.train_csv, Config.working_dir, load_cached_data=True
    )

    # Train Dataset
    train_dataset = HotelDataset(
        Config.train_csv,
        Config.input_dir,
        label_encoder=label_encoder,
        transform=get_transforms(Config.image_size, mode="train"),
        debug=Config.debug,
    )

    # Validation Dataset
    val_dataset = HotelDataset(
        Config.val_csv,
        Config.input_dir,
        label_encoder=label_encoder,
        transform=get_transforms(Config.image_size, mode="val"),
        debug=Config.debug,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print(f"\n=== Initializing Model: {Config.model_name} ===")
    model = HotelModel(
        num_classes=Config.num_classes,
        model_name=Config.model_name,
        embedding_size=Config.embedding_size,
        scale=Config.scale,
        margin=Config.margin,
        k_subcenters=Config.k_subcenters,
        pretrained=Config.pretrained,
    )
    model.to(Config.device)

    # ---------------------------------------------------------
    # 4. Optimization Setup
    # ---------------------------------------------------------
    criterion = ArcFaceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = len(train_loader) * Config.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n=== Starting Training ===")
    best_map = 0.0
    best_model_path = Config.model_path
    checkpoint_path = os.path.join(Config.working_dir, "checkpoint.pth")

    for epoch in range(1, Config.epochs + 1):
        # Train Step
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, Config.device, epoch
        )

        # Validation Step
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.device)
                labels = labels.to(Config.device)

                # Get embeddings and logits
                embeddings = model(images, labels=None)
                logits = model.head(embeddings, labels=None)

                all_preds.append(logits.cpu())
                all_targets.append(labels.cpu())

        # Compute Epoch Metrics
        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        val_map = mean_average_precision(preds_tensor, targets_tensor, k=5)

        print(
            f"Epoch {epoch}/{Config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAP@5: {val_map:.6f}"
        )

        # Save Checkpoint
        if val_map > best_map:
            best_map = val_map
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_map,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filepath=checkpoint_path,
                best_filepath=best_model_path,
            )
            print(f"  >>> New Best MAP@5: {best_map:.6f} (Saved)")

    # ---------------------------------------------------------
    # 6. Final Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Running Final Evaluation & Failure Analysis ===")

    # Load Best Model
    load_checkpoint(model, best_model_path, device=Config.device)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.device)

            embeddings = model(images, labels=None)
            logits = model.head(embeddings, labels=None)

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # 1. Final Validation Metric
    final_metric = mean_average_precision(logits_tensor, targets_tensor, k=5)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    # Calculate per-sample scores
    _, topk_indices = logits_tensor.topk(5, dim=1, largest=True, sorted=True)

    sample_scores = []
    max_confidences = []

    targets_numpy = targets_tensor.numpy()
    topk_numpy = topk_indices.numpy()

    for i in range(len(targets_numpy)):
        target = targets_numpy[i]
        preds = topk_numpy[i]

        # Calculate reciprocal rank score
        score = 0.0
        if target in preds:
            # np.where returns tuple
            rank = np.where(preds == target)[0][0]
            score = 1.0 / (rank + 1)
        sample_scores.append(score)

        # Confidence (Max Logit)
        max_confidences.append(logits_tensor[i].max().item())

    sample_scores = np.array(sample_scores)
    max_confidences = np.array(max_confidences)

    # Calculate Class Frequency Correlation
    # Load training data to get class counts
    train_df = pd.read_csv(Config.train_csv)
    train_hotel_ids = train_df["hotel_id"].values

    # Transform to labels
    train_labels = label_encoder.transform(train_hotel_ids)

    # Count occurrences of each label in training set
    label_counts = np.bincount(train_labels, minlength=len(label_encoder.classes_))

    # Get the training frequency for each validation sample's target class
    val_sample_freqs = label_counts[targets_numpy]

    # Compute correlations
    corr_freq_score = pearson_corr(val_sample_freqs, sample_scores)
    corr_conf_score = pearson_corr(max_confidences, sample_scores)

    print("Failure Analysis Results:")
    print(f"  Correlation (Class Frequency vs Error Magnitude): {corr_freq_score:.6f}")
    print(f"  Correlation (Model Confidence vs Error Magnitude): {corr_conf_score:.6f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.5589516758918762

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = HotelDataset(
            Config.test_csv,
            Config.input_dir,
            is_test=True,
            transform=get_transforms(Config.image_size, mode="test"),
            debug=Config.debug,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        generate_submission(
            test_loader, model, label_encoder, Config.device, Config.submission_path
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
