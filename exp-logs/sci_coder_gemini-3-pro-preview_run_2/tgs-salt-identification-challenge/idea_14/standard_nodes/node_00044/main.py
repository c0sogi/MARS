import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.utils import set_seed, calculate_iou_map
from library.model import WideLinkNetResNet34
from library.dataset import SaltDataset
from library.losses import CombinedLoss
from library.engine import Trainer, validate
from library.inference import predict_proba, optimize_threshold, generate_submission_csv

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 20
LR = 1e-4
WORKING_DIR = "./working"
PSEUDO_MASK_DIR = os.path.join(WORKING_DIR, "pseudo_masks")
os.makedirs(PSEUDO_MASK_DIR, exist_ok=True)
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Load Metadata
    print("\n--- Loading Metadata ---")
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Calculate depth statistics from training set
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std()
    depth_stats = (depth_mean, depth_std)
    print(f"Depth Stats - Mean: {depth_mean:.2f}, Std: {depth_std:.2f}")

    # 3. Stage 1: Robust Supervised Training
    print("\n--- Stage 1: Robust Supervised Training ---")
    # Dataset with Bernoulli Masking (p=0.5)
    train_dataset_s1 = SaltDataset(
        mode="train", df=train_df, depth_stats=depth_stats, bernoulli_mask_prob=0.5
    )
    val_dataset = SaltDataset(
        mode="val", df=val_df, depth_stats=depth_stats, bernoulli_mask_prob=0.0
    )

    train_loader_s1 = DataLoader(
        train_dataset_s1,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model_s1 = WideLinkNetResNet34().to(DEVICE)
    optimizer_s1 = torch.optim.AdamW(model_s1.parameters(), lr=LR)
    criterion = CombinedLoss()

    stage1_dir = os.path.join(WORKING_DIR, "stage1")
    trainer_s1 = Trainer(
        model_s1, optimizer_s1, criterion, DEVICE, patience=6, save_dir=stage1_dir
    )

    trainer_s1.fit(train_loader_s1, val_loader, epochs=STAGE1_EPOCHS)

    # Load best Stage 1 model
    best_s1_path = os.path.join(stage1_dir, "best_model.pth")
    model_s1.load_state_dict(torch.load(best_s1_path, map_location=DEVICE))

    # 4. Generate Pseudo-Labels
    print("\n--- Generating Pseudo-Labels for Test Set ---")
    # Test dataset (no masks)
    test_dataset = SaltDataset(mode="test", df=test_df, depth_stats=depth_stats)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Predict with force_zero_depth=True (Generalist Mode)
    results = predict_proba(model_s1, test_loader, DEVICE, force_zero_depth=True)
    test_probs = results["predictions"]
    test_ids = results["ids"]

    # Determine threshold from validation set
    print("Optimizing threshold on validation set for pseudo-labeling...")
    # Get val predictions
    val_res_s1 = predict_proba(model_s1, val_loader, DEVICE, force_zero_depth=False)
    pseudo_thresh = optimize_threshold(val_res_s1["masks"], val_res_s1["predictions"])
    print(f"Pseudo-labeling threshold: {pseudo_thresh}")

    # Save pseudo-masks and create DataFrame
    pseudo_rows = []
    for i, img_id in enumerate(test_ids):
        # Binarize
        mask = (test_probs[i] > pseudo_thresh).astype(np.uint8) * 255

        # Save image
        fname = f"{img_id}.png"
        save_path = os.path.join(PSEUDO_MASK_DIR, fname)
        cv2.imwrite(save_path, mask)

        # Create relative path for SaltDataset (hack for ./input root)
        # ./input/../working/pseudo_masks/id.png
        rel_path = os.path.join("..", "working", "pseudo_masks", fname)

        # Get metadata
        row_meta = test_df[test_df["id"] == img_id].iloc[0]

        pseudo_rows.append(
            {
                "id": img_id,
                "z": row_meta["z"],
                "image_path": row_meta["image_path"],
                "mask_path": rel_path,
            }
        )

    pseudo_df = pd.DataFrame(pseudo_rows)
    print(f"Generated {len(pseudo_df)} pseudo-labels.")

    # 5. Stage 2: Retraining with Pseudo-Labels
    print("\n--- Stage 2: Retraining on Combined Data ---")
    # Combine data
    cols = ["id", "z", "image_path", "mask_path"]
    combined_df = pd.concat([train_df[cols], pseudo_df[cols]], ignore_index=True)

    # Combined Dataset
    combined_dataset = SaltDataset(
        mode="train", df=combined_df, depth_stats=depth_stats, bernoulli_mask_prob=0.5
    )
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Fresh Model
    model_s2 = WideLinkNetResNet34().to(DEVICE)
    optimizer_s2 = torch.optim.AdamW(model_s2.parameters(), lr=LR)

    stage2_dir = os.path.join(WORKING_DIR, "stage2")
    trainer_s2 = Trainer(
        model_s2, optimizer_s2, criterion, DEVICE, patience=6, save_dir=stage2_dir
    )

    trainer_s2.fit(combined_loader, val_loader, epochs=STAGE2_EPOCHS)

    # Load best Stage 2 model
    best_s2_path = os.path.join(stage2_dir, "best_model.pth")
    model_s2.load_state_dict(torch.load(best_s2_path, map_location=DEVICE))

    # 6. Final Evaluation
    print("\n--- Final Evaluation ---")
    # Get predictions on validation set
    val_results = predict_proba(model_s2, val_loader, DEVICE, force_zero_depth=False)
    val_probs = val_results["predictions"]
    val_masks = val_results["masks"]
    val_ids = val_results["ids"]

    # Optimize threshold
    best_thresh = optimize_threshold(val_masks, val_probs)

    # Calculate Final Metric
    val_preds_bin = (val_probs > best_thresh).astype(np.uint8)
    final_metric = calculate_iou_map(val_masks, val_preds_bin)

    print(f"Final Validation Metric: {final_metric:.10f}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate IoU per image
    ious = []
    for i in range(len(val_masks)):
        t = val_masks[i]
        p = val_preds_bin[i]

        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()

        if union == 0:
            iou = 1.0  # Both empty
        else:
            iou = inter / union
        ious.append(iou)

    ious = np.array(ious)
    errors = 1.0 - ious

    # Align with metadata
    val_meta_indexed = val_df.set_index("id")
    depths = []
    coverages = []

    for vid in val_ids:
        row = val_meta_indexed.loc[vid]
        depths.append(row["z"])
        coverages.append(row["salt_coverage"])

    corr_depth = np.corrcoef(errors, depths)[0, 1]
    corr_cov = np.corrcoef(errors, coverages)[0, 1]

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 8. Submission
    if final_metric > 0.7985:
        print("\n--- Generating Submission ---")
        # Predict on Test Set (Force Depth 0)
        test_results = predict_proba(
            model_s2, test_loader, DEVICE, force_zero_depth=True
        )
        test_probs = test_results["predictions"]
        test_ids = test_results["ids"]

        generate_submission_csv(
            test_ids, test_probs, best_thresh, output_path=SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric {final_metric:.4f} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
