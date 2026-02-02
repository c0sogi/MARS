import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    compute_dice_coefficient,
    rle_decode,
)
from library.data import get_dataloaders
from library.model import build_model
from library.losses import CurriculumLoss
from library.engine import fit, validate_3d, inverse_transform_slice


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error patterns.
    Computes correlations between error magnitude (1 - Dice) and metadata features.
    """
    print("\n=== Performing Failure Analysis ===")
    model.eval()

    results = []

    # We will analyze at the slice level for granular correlation
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)  # (B, C, H, W) - Padded
            ids = batch["id"]
            pad_infos = batch["padding_info"]

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            preds = torch.sigmoid(outputs)

            # Convert to numpy
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Iterate through batch
            for i in range(len(ids)):
                # Parse metadata from ID
                # ID format: caseXXX_dayYY_slice_ZZZZ
                parts = ids[i].split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                # Extract padding info
                p_info = {
                    k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i]
                    for k, v in pad_infos.items()
                }

                # We calculate metrics on the padded space for this correlation analysis
                # to avoid overhead of resizing everything, as relative error holds.
                # However, for area calculation, we should consider the mask content.

                # Calculate Dice per class
                row_data = {"id": ids[i], "slice_num": slice_num}

                total_error = 0
                total_area = 0

                for c_idx, cls_name in enumerate(Config.CLASSES):
                    y_true = masks_np[i, c_idx]
                    y_pred = (preds_np[i, c_idx] > Config.MASK_THRESHOLD).astype(
                        np.float32
                    )

                    dice = compute_dice_coefficient(y_true, y_pred)
                    area = np.sum(y_true)

                    row_data[f"dice_{cls_name}"] = dice
                    row_data[f"area_{cls_name}"] = area

                    total_error += 1.0 - dice
                    total_area += area

                row_data["mean_error"] = total_error / Config.NUM_CLASSES
                row_data["total_area"] = total_area
                results.append(row_data)

    df_analysis = pd.DataFrame(results)

    # Normalize slice number per case (approximate) to get relative position
    # We don't have max slice per case readily available in this loop without lookups,
    # but we can infer it from the dataframe.
    max_slices = df_analysis.groupby(
        df_analysis["id"].apply(lambda x: "_".join(x.split("_")[:2]))
    )["slice_num"].transform("max")
    df_analysis["rel_position"] = df_analysis["slice_num"] / max_slices

    # Calculate Correlations
    corr_area = df_analysis["mean_error"].corr(df_analysis["total_area"])
    corr_pos = df_analysis["mean_error"].corr(df_analysis["rel_position"])

    print(f"Correlation (Error vs Mask Area): {corr_area:.4f}")
    print(f"Correlation (Error vs Slice Position): {corr_pos:.4f}")

    # Interpretation
    if abs(corr_area) > 0.3:
        print("-> Significant relationship between organ size and error rate.")
    if abs(corr_pos) > 0.3:
        print("-> Significant relationship between slice depth and error rate.")

    return df_analysis


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\n=== Generating Submission ===")
    model.eval()

    # Store predictions in a lookup dictionary: key=(id, class), value=rle
    pred_lookup = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)
            ids = batch["id"]
            pad_infos = batch["padding_info"]

            # Predict
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            preds = torch.sigmoid(outputs).cpu().numpy()

            # Process batch
            for i, sample_id in enumerate(ids):
                # Extract padding info
                p_info = {
                    k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i]
                    for k, v in pad_infos.items()
                }

                # Inverse Transform (Crop & Resize)
                # Returns (C, H_orig, W_orig)
                restored_pred = inverse_transform_slice(preds[i], p_info)

                # Threshold
                binary_mask = (restored_pred > Config.MASK_THRESHOLD).astype(np.uint8)

                # Encode each class
                for c_idx, cls_name in enumerate(Config.CLASSES):
                    rle = rle_encode(binary_mask[c_idx])
                    pred_lookup[(sample_id, cls_name)] = rle

    # Load Template (Cite debug_lesson_1)
    # Using sample_submission.csv to ensure correct row order and structure
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        df_sub = pd.read_csv(sample_sub_path)

        # Helper to retrieve prediction
        def get_pred(row):
            key = (row["id"], row["class"])
            return pred_lookup.get(key, "")

        df_sub["predicted"] = df_sub.apply(get_pred, axis=1)
    else:
        # Fallback if sample_submission doesn't exist
        print(
            "Warning: sample_submission.csv not found. Constructing from predictions."
        )
        submission_rows = []
        for (sid, cls), rle in pred_lookup.items():
            submission_rows.append({"id": sid, "class": cls, "predicted": rle})
        df_sub = pd.DataFrame(submission_rows)
        df_sub = df_sub[["id", "class", "predicted"]]

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(df_sub)}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # Increased resolution requires careful time management; 8 epochs should suffice for U-Net++ (Cite Lesson 20)
    # Disabled Boundary Loss (Warmup >= Epochs) to prioritize stability and avoid resolution penalties (Cite Lesson 27)
    config = Config()
    config.EPOCHS = 8
    config.WARMUP_EPOCHS = 8
    config.DEBUG = False  # Use full dataset for best performance

    print(f"Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Warmup: {config.WARMUP_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Backbone: {config.BACKBONE}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 3. Model & Optimization
    print("Building model...")
    model = build_model(config)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)

    loss_fn = CurriculumLoss(config)

    # 4. Training
    print("Starting training...")
    fit(model, train_loader, val_loader, optimizer, scheduler, loss_fn, config)

    # 5. Final Validation & Analysis
    print("Loading best model for validation...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    # Load weights
    checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.to(config.DEVICE)
    model.eval()

    # Compute Final Metric
    val_metrics = validate_3d(model, val_loader, config.DEVICE)
    final_score = val_metrics["val_score"]

    print(f"Final Validation Metric: {final_score:.10f}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, config.DEVICE)

    # 6. Submission
    threshold = 0.5432270206
    if final_score > threshold:
        print(
            f"Validation score ({final_score:.4f}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, config.DEVICE, config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score ({final_score:.4f}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
