import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import HRNetCenterNet
from library.engine import fit, load_val_gt_info, decode_outputs
from library.utils import calc_f1_score, get_affine_transform, affine_transform
from library.inference import generate_submission


def main():
    # 1. Setup and Configuration
    Config.setup()
    Config.seed_everything(Config.SEED)

    # Adjust hyperparameters for a fast baseline execution
    # Reducing epochs to ensure completion within strict time limits
    Config.NUM_EPOCHS = 10

    # 2. Data Loading
    # We use the full dataset but limited epochs.
    train_dataset = KuzushijiDataset(split="train")
    val_dataset = KuzushijiDataset(split="val")

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
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    device = Config.DEVICE
    model = HRNetCenterNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    # Using the provided engine.fit function
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=5,
    )

    # 5. Validation and Failure Analysis
    # Load the best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    # Load Ground Truths for Validation
    val_gt_data = load_val_gt_info(Config.VAL_METADATA_PATH)

    all_preds_for_metric = []
    all_gts_for_metric = []

    # Lists for failure analysis
    fa_errors = []
    fa_widths = []
    fa_heights = []
    fa_ann_counts = []

    print("Performing validation inference and failure analysis...")

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            img_ids = batch["image_id"]

            # Inference
            hm, wh, reg = model(imgs)
            scores, clses, xs, ys = decode_outputs(hm, wh, reg, K=Config.MAX_PREDS)

            # Process each image in the batch
            for b in range(len(img_ids)):
                img_id = img_ids[b]

                if img_id not in val_gt_data:
                    continue

                # Get GT info
                orig_h, orig_w = val_gt_data[img_id]["size"]
                gts = val_gt_data[img_id]["gts"]

                # Inverse Transform to map predictions back to original image space
                trans_inv = get_affine_transform(
                    (orig_h, orig_w), Config.INPUT_SIZE, inverse=True
                )

                # Filter and Transform Predictions
                valid_mask = scores[b] > Config.CONF_THRESHOLD
                v_scores = scores[b][valid_mask]
                v_clses = clses[b][valid_mask]
                v_xs = xs[b][valid_mask]
                v_ys = ys[b][valid_mask]

                img_preds = []
                for k in range(len(v_scores)):
                    pt = affine_transform([v_xs[k], v_ys[k]], trans_inv)
                    px = min(max(0, pt[0]), orig_w - 1)
                    py = min(max(0, pt[1]), orig_h - 1)

                    img_preds.append(
                        {
                            "point": (px, py),
                            "label": v_clses[k],
                            "score": float(v_scores[k]),
                        }
                    )

                # Store for global metric
                all_preds_for_metric.append(img_preds)
                all_gts_for_metric.append(gts)

                # --- Failure Analysis Calculation (Per Image) ---
                # Calculate FP and FN for this image

                # Sort predictions by score
                p_sorted = sorted(img_preds, key=lambda x: x["score"], reverse=True)
                gt_matched = [False] * len(gts)

                tp_count = 0
                fp_count = 0

                for p in p_sorted:
                    p_x, p_y = p["point"]
                    p_label = p["label"]

                    match_found = False
                    for i, gt in enumerate(gts):
                        if gt_matched[i]:
                            continue

                        gt_label = gt["label"]
                        gt_x, gt_y, gt_w, gt_h = gt["box"]

                        if p_label != gt_label:
                            continue

                        # Check if point inside box
                        if (gt_x <= p_x <= gt_x + gt_w) and (
                            gt_y <= p_y <= gt_y + gt_h
                        ):
                            gt_matched[i] = True
                            match_found = True
                            break

                    if match_found:
                        tp_count += 1
                    else:
                        fp_count += 1

                fn_count = sum(1 for m in gt_matched if not m)
                error_magnitude = fp_count + fn_count

                fa_errors.append(error_magnitude)
                fa_widths.append(orig_w)
                fa_heights.append(orig_h)
                fa_ann_counts.append(len(gts))

    # Calculate and Print Final Metric
    final_f1 = calc_f1_score(all_preds_for_metric, all_gts_for_metric)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis Correlations
    fa_df = pd.DataFrame(
        {
            "error": fa_errors,
            "width": fa_widths,
            "height": fa_heights,
            "num_anns": fa_ann_counts,
        }
    )

    print("Failure Analysis Correlations (Error Magnitude vs Features):")
    if len(fa_df) > 0:
        corr_width = fa_df["error"].corr(fa_df["width"])
        corr_height = fa_df["error"].corr(fa_df["height"])
        corr_anns = fa_df["error"].corr(fa_df["num_anns"])

        print(f"Correlation with Image Width: {corr_width}")
        print(f"Correlation with Image Height: {corr_height}")
        print(f"Correlation with Number of Annotations: {corr_anns}")
    else:
        print("Not enough data for failure analysis.")

    # 6. Submission Generation
    # Condition: F1 > 0.7679033467456621
    TARGET_THRESHOLD = 0.7679033467456621

    if final_f1 > TARGET_THRESHOLD:
        print(
            f"Validation F1 ({final_f1}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"Validation F1 ({final_f1}) did not exceed threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
