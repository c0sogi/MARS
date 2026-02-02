import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import SIIMDataset, get_transforms
from library.model import get_model
from library.engine import Engine
from library.utils import seed_everything, collate_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, data_loader, device):
    """
    Calculates loss per image and correlates it with metadata features.
    """
    print("\nPerforming Failure Analysis...")

    # To get loss, model must be in train mode
    model.train()

    results = []

    with torch.no_grad():
        for images, targets, image_ids in data_loader:
            images = list(img.to(device) for img in images)
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass to get losses
            # Note: torchvision returns a dict of losses. We sum them.
            # However, standard forward() averages over batch.
            # We need per-image loss. We process one by one effectively or assume batch average approximates.
            # For accurate per-image analysis, we pass images one by one or extract unreduced loss if possible.
            # The standard model reduces loss. We will iterate batch items manually to get per-item loss.

            for i in range(len(images)):
                single_img = [images[i]]
                single_target = [targets_gpu[i]]

                loss_dict = model(single_img, single_target)
                total_loss = sum(loss for loss in loss_dict.values()).item()

                # Extract features from CPU target
                tgt = targets[i]
                num_boxes = len(tgt["boxes"])

                if num_boxes > 0:
                    areas = (tgt["boxes"][:, 2] - tgt["boxes"][:, 0]) * (
                        tgt["boxes"][:, 3] - tgt["boxes"][:, 1]
                    )
                    avg_area = areas.mean().item()
                else:
                    avg_area = 0.0

                results.append(
                    {"loss": total_loss, "num_boxes": num_boxes, "avg_area": avg_area}
                )

    df = pd.DataFrame(results)

    # Calculate correlations
    if len(df) > 0:
        corr_boxes = df["loss"].corr(df["num_boxes"])
        corr_area = df["loss"].corr(df["avg_area"])

        print("Correlation between Error (Loss) and Input Features:")
        print(f"  Loss vs Num Boxes: {corr_boxes:.4f}")
        print(f"  Loss vs Avg Box Area: {corr_area:.4f}")
    else:
        print("No data for failure analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Use full dataset for valid metric calculation
    train_dataset = SIIMDataset(
        train_df,
        mode="train",
        transforms=get_transforms("train"),
        limit_size=None,
    )

    val_dataset = SIIMDataset(
        val_df, mode="val", transforms=get_transforms("val"), limit_size=None
    )

    test_dataset = SIIMDataset(test_df, mode="test", transforms=get_transforms("test"))

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # 3. Model Initialization
    model = get_model(num_classes=Config.NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )

    # 4. Training
    engine = Engine(model, device, optimizer, lr_scheduler)

    # Use configured epochs for full training
    EPOCHS = Config.EPOCHS
    best_model_path = engine.fit_model(
        train_loader, val_loader, epochs=EPOCHS, patience=5
    )

    # 5. Validation Assessment
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Calculate mAP
    print("Running validation inference for mAP calculation...")
    map_score = engine.evaluate_map(val_loader)
    print(f"Final Validation Metric: {map_score}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 6. Inference and Submission
    if map_score > 0.19051633522746228:
        engine.inference(test_loader, best_model_path)
    else:
        print("Validation metric did not meet threshold. Skipping submission.")


if __name__ == "__main__":
    main()
