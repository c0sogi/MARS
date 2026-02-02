import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
import warnings
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import WhaleDataset, get_transforms, get_class_mapping
from library.model import WhaleDenseNet
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Train
    print("Starting training pipeline...")
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Metrics
    print("Loading best SWA model for final evaluation...")
    device = Config.DEVICE

    # Initialize Model Structure
    model = WhaleDenseNet(
        backbone_name=Config.BACKBONE,
        pretrained=False,
        embedding_size=Config.EMBEDDING_SIZE,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Load Weights: Prefer SWA Final, fallback to Model Best
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "swa_model_final.pth.tar")
    if not os.path.exists(checkpoint_path):
        print("SWA checkpoint not found, falling back to model_best...")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle state dict keys (strip 'module.' if present from AveragedModel wrapper)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        elif k.startswith("n_averaged"):
            continue
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Validation Loop
    val_dataset = WhaleDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []
    all_image_ids = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels, image_ids in val_loader:
            images = images.to(device)

            # TTA: Original + Flip
            logits = model(images, labels=None)
            if Config.TTA_FLIP:
                logits_flip = model(torch.flip(images, dims=[3]), labels=None)
                logits = (logits + logits_flip) / 2.0

            _, top_indices = torch.topk(logits, k=5, dim=1)

            all_preds.append(top_indices.cpu())
            all_targets.append(labels.cpu())
            all_image_ids.extend(image_ids)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    val_map5 = calculate_map5(all_preds, all_targets)
    print(f"Final Validation Metric: {val_map5}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    val_df = val_dataset.df.set_index("Image")

    failures = []

    # Convert tensors to numpy for iteration
    targets_np = all_targets.numpy()
    preds_np = all_preds.numpy()

    for i, img_id in enumerate(all_image_ids):
        target = targets_np[i]
        pred_top5 = preds_np[i]

        # Error: 1 if target NOT in top 5, else 0
        is_error = 1 if target not in pred_top5 else 0

        # Get metadata features
        try:
            rel_path = val_df.loc[img_id, "file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                # Simple intensity (grayscale mean)
                intensity = img.mean()

                failures.append(
                    {
                        "error": is_error,
                        "width": w,
                        "height": h,
                        "aspect_ratio": w / h if h > 0 else 0,
                        "intensity": intensity,
                    }
                )
        except Exception:
            continue

    if len(failures) > 0:
        fail_df = pd.DataFrame(failures)
        correlations = fail_df.corr()["error"].drop("error")
        print("Correlation between Error and Input Features:")
        print(correlations)
    else:
        print("No failure analysis data collected.")

    # 5. Submission
    threshold = 0.6545824094604581
    if val_map5 > threshold:
        print(
            f"Metric ({val_map5}) > Threshold ({threshold}). Generating submission..."
        )

        # Get class mapping for decoding
        classes, _ = get_class_mapping()

        test_dataset = WhaleDataset(mode="test", transform=get_transforms("val"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(device)

                # TTA
                logits = model(images, labels=None)
                if Config.TTA_FLIP:
                    logits_flip = model(torch.flip(images, dims=[3]), labels=None)
                    logits = (logits + logits_flip) / 2.0

                _, top_indices = torch.topk(logits, k=5, dim=1)
                top_indices = top_indices.cpu().numpy()

                for i, img_id in enumerate(image_ids):
                    pred_indices = top_indices[i]
                    # Map indices to class names
                    pred_labels = [classes[idx] for idx in pred_indices]
                    pred_str = " ".join(pred_labels)
                    submission_rows.append([img_id, pred_str])

        sub_df = pd.DataFrame(submission_rows, columns=["Image", "Id"])
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(f"Metric ({val_map5}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    main()
