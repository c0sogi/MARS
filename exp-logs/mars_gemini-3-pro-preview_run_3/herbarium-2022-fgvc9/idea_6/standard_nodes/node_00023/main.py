import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import f1_score

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloader
from library.model import HierarchicalEfficientNet
from library.train import train_one_epoch
from library.predict import inference


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for fast baseline execution within time limits
    Config.STAGE_1_EPOCHS = 1
    Config.STAGE_2_EPOCHS = 1

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Subsample training data to ensure execution finishes within 2 hours
    # Using 80,000 samples (~15% of data) for the baseline run
    print(f"Original training samples: {len(df_train)}")
    df_train = df_train.sample(n=80000, random_state=Config.SEED).reset_index(drop=True)
    print(f"Subsampled training samples: {len(df_train)}")

    # -------------------------------------------------------------------------
    # 3. Initialize Model
    # -------------------------------------------------------------------------
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = HierarchicalEfficientNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # -------------------------------------------------------------------------
    # 4. Stage 1 Training (Feature Learning)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"STAGE 1: Training at {Config.STAGE_1_RES}x{Config.STAGE_1_RES}")
    print("=" * 40)

    train_loader_s1 = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.STAGE_1_BATCH_SIZE,
        image_size=Config.STAGE_1_RES,
        shuffle=True,
    )

    optimizer_s1 = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler_s1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s1,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader_s1),
        epochs=Config.STAGE_1_EPOCHS,
        pct_start=0.1,
    )

    for epoch in range(Config.STAGE_1_EPOCHS):
        loss = train_one_epoch(
            model, train_loader_s1, optimizer_s1, scheduler_s1, device, criterion, epoch
        )
        print(f"Stage 1 Epoch {epoch+1} Loss: {loss:.4f}")

    # Save Stage 1 Checkpoint
    torch.save(model.state_dict(), Config.CHECKPOINT_STAGE_1)
    print("Stage 1 completed.")

    # -------------------------------------------------------------------------
    # 5. Stage 2 Training (Fine-Grained Refinement)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"STAGE 2: Fine-tuning at {Config.STAGE_2_RES}x{Config.STAGE_2_RES}")
    print("=" * 40)

    # Load Stage 1 weights
    model.load_state_dict(torch.load(Config.CHECKPOINT_STAGE_1))

    train_loader_s2 = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.STAGE_2_BATCH_SIZE,
        image_size=Config.STAGE_2_RES,
        shuffle=True,
    )

    optimizer_s2 = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler_s2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s2,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader_s2),
        epochs=Config.STAGE_2_EPOCHS,
        pct_start=0.1,
    )

    for epoch in range(Config.STAGE_2_EPOCHS):
        loss = train_one_epoch(
            model, train_loader_s2, optimizer_s2, scheduler_s2, device, criterion, epoch
        )
        print(f"Stage 2 Epoch {epoch+1} Loss: {loss:.4f}")

    # Save Stage 2 Checkpoint (Best Model for Inference)
    torch.save(model.state_dict(), Config.CHECKPOINT_STAGE_2)
    print("Stage 2 completed.")

    # -------------------------------------------------------------------------
    # 6. Final Validation & Metrics
    # -------------------------------------------------------------------------
    print("\nPerforming Final Validation on full validation set...")

    val_loader = get_dataloader(
        df_val,
        mode="valid",
        batch_size=Config.STAGE_2_BATCH_SIZE,
        image_size=Config.STAGE_2_RES,
        shuffle=False,
    )

    model.eval()
    all_preds = []
    all_labels = []

    # Inference loop for validation
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            labels_species = targets["species"].to(device)

            outputs = model(images)
            # Use Species head for primary metric
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_species.cpu().numpy())

    # Calculate Macro F1
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    print(f"Final Validation Metric: {f1_macro}")

    # -------------------------------------------------------------------------
    # 7. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Add predictions and errors to the validation dataframe
    # Note: df_val order is preserved because shuffle=False in val_loader
    df_val["predicted"] = all_preds
    df_val["ground_truth"] = all_labels
    df_val["error"] = (df_val["predicted"] != df_val["ground_truth"]).astype(int)

    # Sample a subset for feature extraction to keep analysis fast
    # We need to read images to get width/height, which is I/O intensive
    analysis_sample_size = min(1000, len(df_val))
    df_analysis = df_val.sample(n=analysis_sample_size, random_state=Config.SEED).copy()

    stats = []
    print(
        f"Analyzing {analysis_sample_size} validation samples for error correlations..."
    )

    for idx, row in df_analysis.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Get file size
            file_size = os.path.getsize(file_path)

            # Get image dimensions
            img = cv2.imread(file_path)
            if img is not None:
                h, w, _ = img.shape
                stats.append(
                    {
                        "error": row["error"],
                        "file_size": file_size,
                        "width": w,
                        "height": h,
                        "aspect_ratio": w / h if h > 0 else 0,
                    }
                )
        except Exception:
            continue

    if stats:
        df_stats = pd.DataFrame(stats)

        # Calculate correlations
        corr_size = df_stats["error"].corr(df_stats["file_size"])
        corr_width = df_stats["error"].corr(df_stats["width"])
        corr_height = df_stats["error"].corr(df_stats["height"])

        print("Correlation between Error and Input Features:")
        print(f"  File Size: {corr_size:.4f}")
        print(f"  Width:     {corr_width:.4f}")
        print(f"  Height:    {corr_height:.4f}")
    else:
        print("Could not collect statistics for failure analysis.")

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.5930838412243743

    if f1_macro > threshold:
        print(
            f"\nValidation Metric ({f1_macro}) > Threshold ({threshold}). Generating submission..."
        )
        # Call the inference function from library.predict
        # It handles test loading, TTA, and saving to submission.csv
        inference(checkpoint_path=Config.CHECKPOINT_STAGE_2)
    else:
        print(
            f"\nValidation Metric ({f1_macro}) <= Threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
