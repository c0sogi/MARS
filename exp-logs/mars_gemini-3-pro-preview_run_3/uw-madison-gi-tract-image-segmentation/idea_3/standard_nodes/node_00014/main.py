import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import gc

# Import from library
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    keep_largest_component,
    get_dice_score,
    get_3d_hausdorff,
)
from library.dataset import process_metadata, UWMadissonDataset, get_transforms
from library.trainer import Trainer
from library.model import SegmentationModel


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Patch Config for Fast Baseline Execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32  # Ensure efficient usage of A100

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running Fast Baseline with Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    print("Loading Metadata...")
    df_train = process_metadata(Config.TRAIN_CSV, mode="train")
    df_val = process_metadata(Config.VAL_CSV, mode="valid")

    # Subsample training data for fast baseline (20% of data)
    # We group by case to keep scans intact even when subsampling
    cases = df_train["case"].unique()
    np.random.shuffle(cases)
    subset_cases = cases[: int(len(cases) * 0.25)]  # Use 25% of cases
    df_train_subset = df_train[df_train["case"].isin(subset_cases)].reset_index(
        drop=True
    )

    print(
        f"Training on {len(df_train_subset)} slices (subset). Validation on {len(df_val)} slices."
    )

    # Datasets
    train_dataset = UWMadissonDataset(
        df_train_subset, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = UWMadissonDataset(
        df_val, transforms=get_transforms("valid"), mode="valid"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader)

    print("Starting Training...")
    trainer.fit()

    # Free up memory
    del trainer, train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("Loading Best Model for Evaluation...")
    model = SegmentationModel()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("Running Full Validation Inference...")
    volume_data = defaultdict(list)

    # Inference Loop
    with torch.no_grad():
        for data in tqdm(val_loader, disable=True):
            images = data["image"].to(device, dtype=torch.float32)
            masks = data["mask"].cpu().numpy()
            ids = data["id"]

            with torch.amp.autocast("cuda", enabled=Config.MIXED_PRECISION):
                logits = model(images)
                probs = torch.sigmoid(logits)

            probs = probs.cpu().numpy()

            for i in range(len(ids)):
                # ID format: caseXXX_dayYY_slice_ZZZZ
                parts = ids[i].split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                volume_data[case_day].append(
                    {"slice": slice_num, "pred": probs[i], "mask": masks[i]}
                )

    # Compute Metrics & Failure Analysis Data
    case_scores = []

    print("Computing 3D Metrics...")
    for case_day, slices in volume_data.items():
        slices.sort(key=lambda x: x["slice"])

        # Stack: (D, C, H, W)
        preds_stacked = np.stack([s["pred"] for s in slices], axis=0)
        masks_stacked = np.stack([s["mask"] for s in slices], axis=0)

        # Transpose to (C, D, H, W)
        preds_vol = np.transpose(preds_stacked, (1, 0, 2, 3))
        masks_vol = np.transpose(masks_stacked, (1, 0, 2, 3))

        metrics_per_class = []

        for c in range(Config.NUM_CLASSES):
            p_bin = (preds_vol[c] > Config.MASK_THRESHOLD).astype(np.uint8)
            t_bin = (masks_vol[c] > 0.5).astype(np.uint8)

            # Post-processing
            p_bin = keep_largest_component(p_bin)

            d = get_dice_score(p_bin, t_bin)
            h = get_3d_hausdorff(p_bin, t_bin)

            # Score contribution: 0.4*Dice + 0.6*(1-HD)
            class_score = 0.4 * d + 0.6 * (1.0 - h)
            metrics_per_class.append(class_score)

        avg_score = np.mean(metrics_per_class)

        # Collect data for failure analysis
        # Features: Volume Depth (slices), Total Mask Volume (pixels)
        total_mask_pixels = np.sum(masks_vol)
        depth = masks_vol.shape[1]

        case_scores.append(
            {
                "case_day": case_day,
                "score": avg_score,
                "error": 1.0 - avg_score,
                "depth": depth,
                "mask_volume": total_mask_pixels,
            }
        )

    df_scores = pd.DataFrame(case_scores)
    final_metric = df_scores["score"].mean()

    print(f"Final Validation Metric: {final_metric:.18f}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    if not df_scores.empty:
        corr_depth = df_scores["error"].corr(df_scores["depth"])
        corr_vol = df_scores["error"].corr(df_scores["mask_volume"])

        print(f"Correlation (Error vs Scan Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Target Volume): {corr_vol:.4f}")

        worst_cases = df_scores.sort_values("error", ascending=False).head(5)
        print("\nTop 5 Worst Performing Cases:")
        print(
            worst_cases[
                ["case_day", "score", "error", "depth", "mask_volume"]
            ].to_string(index=False)
        )

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    if final_metric > 0.448:
        print("\nMetric passed threshold. Generating Submission...")

        # Load Test Data
        df_test = process_metadata(Config.TEST_CSV, mode="test")
        test_dataset = UWMadissonDataset(
            df_test, transforms=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Test Inference
        test_volume_data = defaultdict(list)

        with torch.no_grad():
            for data in tqdm(test_loader, disable=True):
                images = data["image"].to(device, dtype=torch.float32)
                ids = data["id"]  # List of strings

                with torch.amp.autocast("cuda", enabled=Config.MIXED_PRECISION):
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                probs = probs.cpu().numpy()

                for i in range(len(ids)):
                    # Parse ID to group by case_day
                    # id: caseXXX_dayYY_slice_ZZZZ
                    parts = ids[i].split("_")
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_num = int(parts[3])

                    test_volume_data[case_day].append(
                        {
                            "id": ids[i],
                            "slice": slice_num,
                            "pred": probs[i],  # (C, H, W)
                        }
                    )

        # Process and Format Submission
        submission_rows = []
        classes = Config.CLASSES

        print("Processing 3D Volumes for Submission...")
        for case_day, slices in test_volume_data.items():
            # Sort by slice to build volume
            slices.sort(key=lambda x: x["slice"])

            # Stack predictions: (D, C, H, W)
            preds_stacked = np.stack([s["pred"] for s in slices], axis=0)
            # Transpose to (C, D, H, W)
            preds_vol = np.transpose(preds_stacked, (1, 0, 2, 3))

            # Process each class volume
            processed_vol = np.zeros_like(preds_vol, dtype=np.uint8)

            for c in range(Config.NUM_CLASSES):
                p_bin = (preds_vol[c] > Config.MASK_THRESHOLD).astype(np.uint8)
                # 3D CCA
                p_processed = keep_largest_component(p_bin)
                processed_vol[c] = p_processed

            # Transpose back to (D, C, H, W) to iterate slices
            processed_vol_slices = np.transpose(processed_vol, (1, 0, 2, 3))

            # Map back to IDs and RLE encode
            for idx, s_data in enumerate(slices):
                slice_id = s_data["id"]
                slice_mask = processed_vol_slices[idx]  # (C, H, W)

                for c_idx, class_name in enumerate(classes):
                    rle = rle_encode(slice_mask[c_idx])
                    submission_rows.append(
                        {"id": slice_id, "class": class_name, "predicted": rle}
                    )

        # Save Submission
        sub_df = pd.DataFrame(submission_rows)
        # Ensure column order
        sub_df = sub_df[["id", "class", "predicted"]]
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric:.4f} did not exceed threshold 0.448. Skipping submission."
        )


if __name__ == "__main__":
    main()
