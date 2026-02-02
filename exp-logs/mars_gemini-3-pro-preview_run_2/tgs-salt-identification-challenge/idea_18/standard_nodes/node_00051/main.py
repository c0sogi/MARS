import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import library modules
from library.utils import seed_everything, rle_encode, do_kaggle_metric, unpad_image
from library.model import ResNet34WideLinkNet
from library.dataset import get_dataloaders
from library.loss import BCELovaszLoss
from library.engine import train_ict_epoch, train_adapt_epoch, predict_proba, evaluate

# Configuration
SEED = 42
BATCH_SIZE = 32
LR = 1e-4
EPOCHS_PHASE_1 = 25
EPOCHS_PHASE_2 = 10
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
METADATA_VAL_PATH = "./metadata/val.csv"


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using load_cached_data=True as requested to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True
    )

    # 3. Model Initialization
    model = ResNet34WideLinkNet(num_classes=1, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = BCELovaszLoss(bce_weight=0.5, lovasz_weight=0.5)

    # 4. Phase 1: Internal Consistency Training
    # Trains on both (Img, Depth) and (Img, Zero) to ensure robustness
    for epoch in range(EPOCHS_PHASE_1):
        loss = train_ict_epoch(model, train_loader, optimizer, criterion, device)
        # Silent progress as requested

    # 5. Generate Soft Pseudo-Labels for Test Set
    # We use the Phase 1 model to guess labels for the test set (using Zero depth)
    pseudo_labels = predict_proba(model, test_loader, device, use_tta=True)

    # 6. Phase 2: Soft-Adaptation
    # Fine-tune using Labeled Data + Unlabeled Test Data (with Pseudo Labels)
    # Re-initialize optimizer for fine-tuning could be beneficial, but continuing is also fine.
    # We lower LR slightly for fine-tuning
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR * 0.5

    for epoch in range(EPOCHS_PHASE_2):
        loss = train_adapt_epoch(
            model,
            train_loader,
            test_loader,
            pseudo_labels,
            optimizer,
            criterion,
            device,
        )

    # 7. Validation & Threshold Optimization
    # evaluate() handles TTA and finds the best threshold
    best_score, best_thresh = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {best_score}")

    # 8. Failure Analysis
    # We need to correlate error with depth and salt coverage.
    # First, load validation metadata to get the ground truth attributes.
    val_meta = pd.read_csv(METADATA_VAL_PATH)

    # Get predictions and GT for validation set manually to compute per-image scores
    val_preds_dict = predict_proba(model, val_loader, device, use_tta=True)

    # Map IDs to scores
    errors = []
    depths = []
    coverages = []

    # We need to access the mask from the loader or metadata.
    # Metadata has paths, but we need the actual mask array for metric calc.
    # We can iterate the val_loader to get masks paired with IDs.
    gt_masks = {}
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                _, masks, _, ids = batch
                masks_np = masks.numpy()
                for i, pid in enumerate(ids):
                    # Unpad mask
                    m = unpad_image(masks_np[i, 0], original_size=101)
                    gt_masks[pid] = m

    for pid, row in val_meta.iterrows():
        img_id = row["id"]
        if img_id in val_preds_dict and img_id in gt_masks:
            pred_prob = val_preds_dict[img_id]
            gt_mask = gt_masks[img_id]

            # Calculate score for this single image
            # do_kaggle_metric expects arrays, returns mean.
            # If we pass single image arrays, it returns the score for that image.
            score = do_kaggle_metric(pred_prob, gt_mask, threshold=best_thresh)

            error = 1.0 - score
            errors.append(error)
            depths.append(row["z"])
            coverages.append(row["salt_coverage"])

    # Calculate correlations
    if len(errors) > 1:
        corr_depth, _ = pearsonr(errors, depths)
        corr_cov, _ = pearsonr(errors, coverages)

        print("\nFailure Analysis:")
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 9. Submission
    if best_score > 0.7985:
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # We already have pseudo_labels (probs) for test set from Phase 1,
        # but we should regenerate them using the Phase 2 model for best results.
        test_probs = predict_proba(model, test_loader, device, use_tta=True)

        submission_rows = []
        for pid, prob_map in test_probs.items():
            # Apply optimized threshold
            binary_mask = (prob_map > best_thresh).astype(np.uint8)
            rle = rle_encode(binary_mask)
            submission_rows.append({"id": pid, "rle_mask": rle})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
