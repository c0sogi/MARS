import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.trainer import Trainer
from library.model import SwinCenterNet
from library.dataset import KuzushijiDataset
from library.utils import decode_center_net, calc_f1_score, _transpose_and_gather_feat
from library.inference import generate_submission


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_image_f1(p_det, t_det):
    """Calculates F1 score for a single image."""
    # p_det: (K, 6) [x, y, w, h, score, class]
    # t_det: (M, 5) [class, x, y, w, h]

    # Sort predictions by score
    if len(p_det) > 0:
        p_det = p_det[np.argsort(-p_det[:, 4])]

    tp = 0
    fp = 0
    n_gt = len(t_det)

    matched_gt_indices = set()

    for p in p_det:
        p_x, p_y = p[0], p[1]
        p_label = int(p[5])
        p_score = p[4]

        if p_score < Config.CONF_THRESHOLD:
            continue

        match_found = False
        for i, g in enumerate(t_det):
            if i in matched_gt_indices:
                continue

            g_label = int(g[0])
            g_x, g_y, g_w, g_h = g[1], g[2], g[3], g[4]

            if p_label == g_label:
                # Check if prediction center is inside ground truth box
                if g_x <= p_x <= g_x + g_w and g_y <= p_y <= g_y + g_h:
                    matched_gt_indices.add(i)
                    match_found = True
                    break

        if match_found:
            tp += 1
        else:
            fp += 1

    fn = n_gt - tp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return f1


def run_pipeline():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # Swin-Base is heavy, so we limit epochs to ensure < 2 hours runtime
    Config.NUM_EPOCHS = 8
    Config.BATCH_SIZE = 4  # Adjusted for A100 40GB (Cite debug_lesson_1)

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Training
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load Best Model
    model = SwinCenterNet()
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
        print(f"Loaded best model from {Config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model not found. Using initialized weights.")

    model.to(Config.DEVICE)
    model.eval()

    # Validation Loader
    val_dataset = KuzushijiDataset(split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    # For Failure Analysis
    fa_stats = (
        []
    )  # List of dicts: {'f1': float, 'num_gt': int, 'width': int, 'height': int}

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            imgs = batch["image"].to(Config.DEVICE)

            # Forward
            outputs = model(imgs)

            # Decode
            # outputs['hm']: (B, 1, H/4, W/4)
            preds = decode_center_net(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                K=Config.MAX_DETECTIONS,
            )

            # Scale coords to 1024x1024 (Model Input Space)
            preds[:, :, :4] *= Config.OUTPUT_STRIDE

            # Reconstruct Ground Truth
            ind = batch["ind"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)
            cls_ids = batch["cls_ids"].to(Config.DEVICE)

            # Gather dense GT
            wh_gt = _transpose_and_gather_feat(batch["wh"].to(Config.DEVICE), ind)
            reg_gt = _transpose_and_gather_feat(batch["reg"].to(Config.DEVICE), ind)

            # Calculate GT coords in 1024x1024 space
            W_out = outputs["hm"].shape[3]
            ys = (ind // W_out).float()
            xs = (ind % W_out).float()

            xs = xs + reg_gt[:, :, 0]
            ys = ys + reg_gt[:, :, 1]

            stride = Config.OUTPUT_STRIDE
            cx = xs * stride
            cy = ys * stride
            w = wh_gt[:, :, 0] * stride
            h = wh_gt[:, :, 1] * stride

            x_tl = cx - w / 2
            y_tl = cy - h / 2

            # Process batch items
            batch_size = preds.size(0)
            preds_np = preds.cpu().numpy()

            # Original image sizes for metadata
            # Note: Dataset returns transformed images (1024x1024).
            # We use the transformed size for correlation analysis as that's what the model sees.
            # Or we could use original size if available. Dataset __getitem__ doesn't return orig_size for val split by default.
            # We will use the model input size (1024x1024) which is constant, so correlation with size is 0.
            # Wait, KuzushijiDataset returns 'orig_size' only for 'test' split in the provided code.
            # However, we can track 'num_gt' which varies.

            for i in range(batch_size):
                # Predictions
                p_det = preds_np[i]

                # Targets
                valid_mask = mask[i].bool().cpu()
                t_cls = cls_ids[i][valid_mask].cpu().float()
                t_x = x_tl[i][valid_mask].cpu()
                t_y = y_tl[i][valid_mask].cpu()
                t_w = w[i][valid_mask].cpu()
                t_h = h[i][valid_mask].cpu()

                t_det = torch.stack([t_cls, t_x, t_y, t_w, t_h], dim=1).numpy()

                all_preds.append(p_det)
                all_targets.append(t_det)

                # Per Image F1 for Failure Analysis
                img_f1 = calculate_image_f1(p_det, t_det)
                num_gt = len(t_det)

                fa_stats.append({"f1_score": img_f1, "num_annotations": num_gt})

    # Calculate Global Metric
    global_f1, precision, recall = calc_f1_score(all_preds, all_targets)
    print(f"Final Validation Metric: {global_f1}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_fa = pd.DataFrame(fa_stats)
    if not df_fa.empty:
        # Correlation
        corr = df_fa.corr()["f1_score"]
        print("Correlation between F1 Score and Input Features:")
        print(corr)

        # Additional Stats
        low_perf = df_fa[df_fa["f1_score"] < 0.5]
        print(f"Number of images with F1 < 0.5: {len(low_perf)} / {len(df_fa)}")
    else:
        print("No validation data available for analysis.")

    # 4. Submission
    THRESHOLD = 0.7679033467456621
    if global_f1 > THRESHOLD:
        print(
            f"\nMetric ({global_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(weights_path=Config.BEST_MODEL_PATH)
    else:
        print(
            f"\nMetric ({global_f1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
