import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, map5
from library.dataset import WhaleDataset, get_transforms, get_class_list
from library.model import WhaleDenseNet
from library.engine import fit, generate_submission, eval_fn


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error and image metadata.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    results = []

    # We need the original dataframe to access file paths for metadata
    # The val_loader dataset has the dataframe
    val_df = val_loader.dataset.df
    input_dir = Config.INPUT_DIR

    # Get predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])

            logits_orig = model(images, labels=None)
            logits_flip = model(images_flip, labels=None)

            avg_logits = (logits_orig + logits_flip) / 2.0

            # Get top 1 prediction for error analysis
            _, top_indices = torch.topk(avg_logits, k=1, dim=1)

            all_preds.extend(top_indices.cpu().numpy().flatten().tolist())
            all_targets.extend(labels.numpy().tolist())

    # Calculate metadata and correlate
    # We iterate through the dataframe and match with predictions
    # Note: DataLoader preserves order if shuffle=False (which is default for val)

    meta_stats = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(input_dir, row["file_path"])

        # Read original image for stats
        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Calculate intensity (simple average)
        intensity = np.mean(img) / 255.0

        target = all_targets[idx]
        pred = all_preds[idx]
        is_error = 1 if target != pred else 0

        meta_stats.append(
            {
                "Width": w,
                "Height": h,
                "AspectRatio": w / h,
                "Intensity": intensity,
                "Error": is_error,
            }
        )

    df_analysis = pd.DataFrame(meta_stats)

    if len(df_analysis) > 0:
        correlations = df_analysis.corr()["Error"].drop("Error")
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Could not perform failure analysis (empty dataframe).")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing Data...")
    # Generate class list first (needed for encoding)
    class_list = get_class_list(load_cached_data=True)

    # Datasets
    train_dataset = WhaleDataset(
        csv_file=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        class_list=class_list,
    )

    val_dataset = WhaleDataset(
        csv_file=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        class_list=class_list,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = WhaleDenseNet()
    model.to(device)

    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 5. Training Loop
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # 6. Evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth.tar")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Compute Final Validation Metric
    val_score = eval_fn(val_loader, model, device)
    print(f"Final Validation Metric: {val_score}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.6545824094604581

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = WhaleDataset(
            csv_file=Config.TEST_CSV, mode="test", transform=get_transforms("test")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        generate_submission(test_loader, model, device)
    else:
        print(
            f"\nValidation score ({val_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
