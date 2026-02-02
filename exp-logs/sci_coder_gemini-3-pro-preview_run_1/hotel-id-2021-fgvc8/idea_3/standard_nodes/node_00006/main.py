import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import process_data, HotelDataset, get_transforms
from library.model import HotelRecognitionModel
from library.engine import train_one_epoch, validate, generate_submission


def run():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Configuration for Fast Baseline Execution
    WARMUP_EPOCHS = 1
    FINE_TUNE_EPOCHS = 10
    SUBMISSION_THRESHOLD = 0.5747

    print(f"Device: {device}")

    # 2. Data Loading
    # Load metadata (cached if available)
    train_df, val_df, test_df, class_map = process_data(
        load_cached_data=True, debug=False
    )

    # Note: Using full dataset as subsampling destroys class density in long-tail distributions.
    # Cite solution_lesson_node_00005

    # Initialize Datasets
    train_dataset = HotelDataset(train_df, transform=get_transforms("train"))
    val_dataset = HotelDataset(val_df, transform=get_transforms("valid"))
    test_dataset = HotelDataset(test_df, transform=get_transforms("test"))

    # Initialize DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = HotelRecognitionModel(
        n_classes=Config.NUM_CLASSES,
        model_name=Config.BACKBONE_NAME,
        embedding_dim=Config.EMBEDDING_DIM,
        margin=Config.MARGIN,
        scale=Config.SCALE,
        pretrained=True,
    )
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    # 4. Training Curriculum

    # --- Stage 1: Softmax Warmup ---
    # Train with margin=0 to stabilize feature extractor without metric penalty
    print("\n=== Stage 1: Softmax Warmup ===")
    model.update_margin(0.0)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_map = validate(model, val_loader, criterion, device)
        print(f"Stage 1 Epoch {epoch} - Val MAP@5: {val_map:.5f}")

    # --- Stage 2: Metric Fine-Tuning ---
    # Enable ArcFace margin to enforce intra-class compactness
    print("\n=== Stage 2: Metric Fine-Tuning ===")
    model.update_margin(Config.MARGIN)

    for epoch in range(WARMUP_EPOCHS + 1, WARMUP_EPOCHS + FINE_TUNE_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_map = validate(model, val_loader, criterion, device)
        print(f"Stage 2 Epoch {epoch} - Val MAP@5: {val_map:.5f}")

    # 5. Final Validation
    # Use the final computed metric
    final_metric = val_map
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    model.eval()

    # We need per-sample scores to correlate with metadata
    # Re-running inference on validation set to get granular data
    all_ranks = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Inference (returns raw cosine similarities)
            outputs = model(images, label=None)
            _, indices = torch.topk(outputs, k=5, dim=1)

            indices_np = indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                target = labels_np[i]
                preds = indices_np[i]

                if target in preds:
                    # Rank is 1-based
                    rank = np.where(preds == target)[0][0] + 1
                    all_ranks.append(1.0 / rank)
                else:
                    all_ranks.append(0.0)

    # Create Analysis DataFrame
    val_df_analysis = val_df.copy()
    val_df_analysis["score"] = all_ranks
    val_df_analysis["error"] = 1.0 - val_df_analysis["score"]

    # Feature Engineering for Correlation
    # Prepare features: Chain (numeric), Time components
    val_df_analysis["chain"] = val_df_analysis["chain"].astype(float)

    features_to_correlate = ["chain", "error"]

    if "timestamp" in val_df_analysis.columns:
        # Parse timestamp
        val_df_analysis["ts"] = pd.to_datetime(
            val_df_analysis["timestamp"], errors="coerce"
        )

        # Extract components
        val_df_analysis["hour"] = val_df_analysis["ts"].dt.hour
        val_df_analysis["month"] = val_df_analysis["ts"].dt.month
        val_df_analysis["year"] = val_df_analysis["ts"].dt.year

        features_to_correlate.extend(["hour", "month", "year"])

    # Calculate Correlation
    # Drop rows with NaNs (e.g. invalid timestamps) for correlation calculation
    corr_df = val_df_analysis[features_to_correlate].dropna()
    corr_matrix = corr_df.corr()

    # Extract correlation with error
    error_corr = corr_matrix["error"].drop("error")

    print("Correlation between Error and Input Features:")
    print(error_corr)

    # 7. Submission
    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model=model,
            dataloader=test_loader,
            test_df=test_df,
            class_map=class_map,
            device=device,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
