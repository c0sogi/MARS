import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_boxes_from_mask
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import EfficientNetUnet
from library.loss import HybridLoss
from library.engine import train_one_epoch, evaluate


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 10  # Limit epochs to ensure runtime < 2 hours
    Config.DEBUG = False  # Ensure we run on full data (or cached data)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing Model...")
    model = EfficientNetUnet(
        encoder_name=Config.ENCODER_NAME,
        pretrained=Config.ENCODER_PRETRAINED,
        num_study_classes=Config.NUM_STUDY_CLASSES,
    ).to(device)

    criterion = HybridLoss(
        seg_weight=Config.SEG_LOSS_WEIGHT, cls_weight=Config.CLS_LOSS_WEIGHT
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_map = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_map = evaluate(val_loader, model, criterion, device)

        # Checkpoint
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  [Checkpoint] New best model saved. mAP: {best_map:.6f}")

    # =========================================================================
    # 5. Final Validation & Failure Analysis
    # =========================================================================
    print("\n==== Final Evaluation & Failure Analysis ====")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # A. Final Metric Calculation
    # We run evaluate() again to ensure we print the metric for the loaded best model
    _, final_map = evaluate(val_loader, model, criterion, device)
    print(f"Final Validation Metric: {final_map}")

    # B. Failure Analysis
    # Calculate correlation between Classification Loss (Error Magnitude) and Number of Boxes (Input Feature)
    print("Performing failure analysis...")

    sample_losses = []
    sample_box_counts = []

    # Loss function for analysis (per sample reduction)
    analysis_criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)  # (B, 1, H, W)

            cls_logits, _ = model(images)

            # Calculate per-sample classification loss
            # cls_logits: (B, 4), labels: (B, 4)
            loss_per_sample = analysis_criterion(cls_logits, labels.float())
            # Average over classes to get a single scalar error per image
            loss_per_sample = loss_per_sample.mean(dim=1).cpu().numpy()

            sample_losses.extend(loss_per_sample)

            # Count boxes per sample in Ground Truth
            masks_np = masks.cpu().numpy()
            for i in range(masks_np.shape[0]):
                # masks_np[i, 0] is (H, W)
                boxes = get_boxes_from_mask(masks_np[i, 0], threshold=0.5)
                sample_box_counts.append(len(boxes))

    # Calculate Correlation
    if len(sample_losses) > 1:
        corr, p_value = pearsonr(sample_losses, sample_box_counts)
        print(
            f"Correlation between Error Magnitude (Loss) and Opacity Count: {corr:.6f}"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    THRESHOLD = 0.4729475001

    if final_map > THRESHOLD:
        print(
            f"\nValidation mAP ({final_map}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nValidation mAP ({final_map}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


def generate_submission(model, device):
    """
    Runs inference on test set and generates submission.csv
    """
    # Load Test Loader
    test_loader = get_test_dataloader(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Test Metadata for IDs and Dimensions
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load Cached Dimensions for Rescaling
    if os.path.exists(Config.TEST_CACHE_DIMS):
        test_dims = pd.read_parquet(Config.TEST_CACHE_DIMS)
        # Map image_id -> {width, height}
        dims_map = test_dims.set_index("id")[["width", "height"]].to_dict("index")
    else:
        # Fallback if cache missing (shouldn't happen if get_test_dataloader ran)
        print("Warning: Test dims cache not found. Using default scaling.")
        dims_map = {}

    # Mappings
    # Config.STUDY_LABELS = ["Negative for Pneumonia", "Typical Appearance", "Indeterminate Appearance", "Atypical Appearance"]
    # Submission requires: negative, typical, indeterminate, atypical
    label_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    results = []
    model.eval()

    current_idx = 0

    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Forward Pass
            cls_logits, seg_logits = model(images)

            # Probabilities
            cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()

            for i in range(batch_size):
                # Get metadata for current image
                row = df_test.iloc[current_idx]
                image_id = row["image_id"]
                study_id = row["study_id"]

                # Get original dimensions
                if image_id in dims_map:
                    orig_w = dims_map[image_id]["width"]
                    orig_h = dims_map[image_id]["height"]
                else:
                    orig_w, orig_h = 1000, 1000  # Fallback

                # --- 1. Study Prediction ---
                # Get class with highest confidence
                best_cls_idx = np.argmax(cls_probs[i])
                best_cls_conf = cls_probs[i][best_cls_idx]
                best_cls_name = Config.ID2LABEL[best_cls_idx]
                pred_label_str = label_map[best_cls_name]

                # Format: "label confidence 0 0 1 1"
                study_pred_string = f"{pred_label_str} {best_cls_conf:.6f} 0 0 1 1"

                results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_pred_string}
                )

                # --- 2. Image Prediction ---
                # Logic: If study is Negative, predict "none"
                if best_cls_name == "Negative for Pneumonia":
                    image_pred_string = "none 1 0 0 1 1"
                else:
                    # Extract boxes from mask
                    mask = seg_probs[i, 0]
                    boxes = get_boxes_from_mask(mask, threshold=0.5)

                    if not boxes:
                        image_pred_string = "none 1 0 0 1 1"
                    else:
                        box_strings = []
                        for box in boxes:
                            # box is [xmin, ymin, xmax, ymax] in 512x512
                            x1, y1, x2, y2 = box

                            # Rescale to original dimensions
                            scale_x = orig_w / Config.IMG_SIZE
                            scale_y = orig_h / Config.IMG_SIZE

                            rx1 = x1 * scale_x
                            ry1 = y1 * scale_y
                            rx2 = x2 * scale_x
                            ry2 = y2 * scale_y

                            # Calculate confidence (mean probability in box area)
                            # Slice using integer indices on the 512 mask
                            roi = mask[int(y1) : int(y2), int(x1) : int(x2)]
                            conf = np.mean(roi) if roi.size > 0 else 0.0

                            box_strings.append(
                                f"opacity {conf:.4f} {rx1:.2f} {ry1:.2f} {rx2:.2f} {ry2:.2f}"
                            )

                        image_pred_string = " ".join(box_strings)

                results.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_string}
                )

                current_idx += 1

    # Save Submission
    submission_df = pd.DataFrame(results)
    # Remove duplicates (multiple images per study -> multiple study rows generated)
    # We keep the first prediction for each ID
    submission_df = submission_df.drop_duplicates(subset="id", keep="first")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
