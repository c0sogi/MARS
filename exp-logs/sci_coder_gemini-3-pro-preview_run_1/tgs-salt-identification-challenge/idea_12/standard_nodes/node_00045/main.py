import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm

# Import from library
from library.config import Config
from library.utils import set_seed, rle_encode, compute_map_batch
from library.dataset import get_dataloaders
from library.model_components import SaltUNet
from library.losses import ConsistentCompoundLoss
from library.engine import train_one_epoch, validate


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-image mAP and correlates with metadata.
    """
    model.eval()

    all_maps = []
    all_depths = []
    all_coverages = (
        []
    )  # We need to calculate coverage from masks since loader doesn't return it directly

    orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE
    thresholds = np.array(Config.IOU_THRESHOLDS)

    print("\n--- Running Failure Analysis ---")

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(device)
            depths_gpu = depths.to(device)

            # Inference
            outputs = model(images, depths_gpu)
            if isinstance(outputs, list):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs)

            # Crop to original size
            h, w = probs.shape[2], probs.shape[3]
            start_h = (h - orig_h) // 2
            start_w = (w - orig_w) // 2

            probs_cropped = probs[
                :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
            ]
            masks_cpu = masks.cpu()
            masks_cropped = masks_cpu[
                :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
            ]

            # Binarize
            preds_np = (probs_cropped > 0.5).float().cpu().numpy().squeeze(1)
            targets_np = masks_cropped.numpy().squeeze(1)

            # Calculate mAP per image manually to get individual scores
            # preds_np: (B, H, W), targets_np: (B, H, W)

            # Flatten spatial dims for IoU calc
            preds_flat = preds_np.reshape(preds_np.shape[0], -1) > 0
            targets_flat = targets_np.reshape(targets_np.shape[0], -1) > 0

            intersection = (preds_flat & targets_flat).sum(axis=1)
            union = (preds_flat | targets_flat).sum(axis=1)

            iou = np.ones(preds_np.shape[0], dtype=np.float32)
            mask_union = union > 0
            iou[mask_union] = intersection[mask_union] / union[mask_union]

            # Compare to thresholds: (B, 1) > (1, T) -> (B, T)
            matches = iou[:, None] > thresholds[None, :]
            ap_per_image = matches.mean(axis=1)

            all_maps.extend(ap_per_image.tolist())
            all_depths.extend(depths.numpy().tolist())

            # Calculate coverage for correlation (pixel count / total pixels)
            covs = targets_flat.sum(axis=1) / (orig_h * orig_w)
            all_coverages.extend(covs.tolist())

    # Calculate Correlations
    # Error = 1.0 - mAP
    errors = 1.0 - np.array(all_maps)
    depths_arr = np.array(all_depths)
    covs_arr = np.array(all_coverages)

    # Handle constant arrays to avoid warnings
    if np.std(errors) == 0:
        corr_depth = 0.0
        corr_cov = 0.0
    else:
        corr_depth, _ = pearsonr(errors, depths_arr)
        corr_cov, _ = pearsonr(errors, covs_arr)

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    return np.mean(all_maps)


def main():
    # 1. Setup
    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(
        f"  Epochs: {Config.EPOCHS} ({Config.CYCLES} cycles of {Config.EPOCHS_PER_CYCLE})"
    )
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=True
    )

    # 3. Model Initialization
    model = SaltUNet().to(Config.DEVICE)

    criterion = ConsistentCompoundLoss().to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
    )

    # 4. Training Loop
    best_map = 0.0
    cycle_best_map = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        # Validate
        val_map = validate(model, val_loader, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        # Logging
        # print(f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val mAP: {val_map:.5f}")

        # Save Best Overall
        if val_map > best_map:
            best_map = val_map
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # Save Best per Cycle (for Snapshot Ensemble)
        # Check if we are in a new cycle to reset tracker
        # Cycles are 1-indexed: 1, 2, 3
        cycle_idx = (epoch - 1) // Config.EPOCHS_PER_CYCLE + 1
        cycle_epoch = (epoch - 1) % Config.EPOCHS_PER_CYCLE + 1

        if cycle_epoch == 1:
            cycle_best_map = 0.0

        if val_map > cycle_best_map:
            cycle_best_map = val_map
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{cycle_idx}.pth"),
            )

    # 5. Final Validation & Failure Analysis
    print("\n--- Training Complete ---")

    # Load best model for final evaluation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    # Run Failure Analysis (calculates correlations and returns exact metric)
    final_metric = run_failure_analysis(model, val_loader, Config.DEVICE)

    print(f"Final Validation Metric: {final_metric:.10f}")

    # 6. Submission Logic
    if final_metric > 0.833:
        print("\n--- Generating Submission ---")

        # Load Ensemble Models
        ensemble_models = []
        # Try to load specified cycles
        for c in Config.ENSEMBLE_CYCLES:
            path = os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{c}.pth")
            if os.path.exists(path):
                m = SaltUNet().to(Config.DEVICE)
                m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
                m.eval()
                ensemble_models.append(m)

        # Fallback to best model if no ensemble models found (or if cycles didn't complete)
        if not ensemble_models:
            ensemble_models.append(model)

        print(f"Ensembling {len(ensemble_models)} models.")

        submission_data = []
        orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE

        with torch.no_grad():
            for images, depths, ids in tqdm(
                test_loader, desc="Inference", disable=True
            ):
                images = images.to(Config.DEVICE)
                depths = depths.to(Config.DEVICE)

                # Accumulate probabilities
                avg_probs = torch.zeros(
                    (images.size(0), 1, Config.IMG_SIZE, Config.IMG_SIZE),
                    device=Config.DEVICE,
                )

                for m in ensemble_models:
                    # Standard
                    out = m(images, depths)
                    if isinstance(out, list):
                        out = out[0]
                    probs = torch.sigmoid(out)

                    # TTA: Horizontal Flip
                    if Config.TTA_FLIP:
                        images_flip = torch.flip(images, [3])
                        out_flip = m(images_flip, depths)
                        if isinstance(out_flip, list):
                            out_flip = out_flip[0]
                        probs_flip = torch.sigmoid(out_flip)
                        probs_flip = torch.flip(probs_flip, [3])
                        probs = (probs + probs_flip) / 2.0

                    avg_probs += probs

                avg_probs /= len(ensemble_models)

                # Crop to 101x101
                h, w = avg_probs.shape[2], avg_probs.shape[3]
                start_h = (h - orig_h) // 2
                start_w = (w - orig_w) // 2

                probs_cropped = avg_probs[
                    :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
                ]

                # Threshold and Encode
                preds = (probs_cropped > 0.5).cpu().numpy()

                for i in range(len(ids)):
                    rle = rle_encode(preds[i, 0])
                    submission_data.append([ids[i], rle])

        # Save
        sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Validation metric {final_metric:.5f} did not meet threshold 0.833. Skipping submission."
        )


if __name__ == "__main__":
    main()
