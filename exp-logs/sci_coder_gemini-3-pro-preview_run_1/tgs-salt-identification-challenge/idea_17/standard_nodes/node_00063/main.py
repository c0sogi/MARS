import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# -------------------------------------------------------------------------
# 1. Configuration Override for Fast Baseline
# -------------------------------------------------------------------------
from library.config import Config

# Override parameters to ensure completion within the time limit.
# We compress the schedule while maintaining the 4-cycle structure to
# generate the necessary checkpoints for the ensemble.
Config.EPOCHS_PER_CYCLE = 12  # Increased to 12 (Total 36 epochs fits in ~7 mins)
Config.CYCLES = 3  # Reduced to 3 cycles to fit time limit
Config.TOTAL_EPOCHS = Config.EPOCHS_PER_CYCLE * Config.CYCLES  # 36 Epochs total
Config.CYCLE_1_END_EPOCH = Config.EPOCHS_PER_CYCLE  # Switch to Lovasz after Cycle 1
Config.QUALITY_GATE_THRESHOLD = 0.05  # Relaxed gating (Cite solution_lesson_node_00062)

# -------------------------------------------------------------------------
# 2. Imports
# -------------------------------------------------------------------------
from library.train import Trainer
from library.evaluate import Evaluator
from library.dataset import get_loaders
from library.utils import do_kaggle_metric, seed_everything


def main():
    print("Starting Runfile Execution...")

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("\n=== Training Phase ===")
    trainer = Trainer(debug=False)
    trainer.train()

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Validation and Failure Analysis Phase ===")
    evaluator = Evaluator()

    # Load validation data
    print("Loading validation data...")
    _, val_loader = get_loaders(debug=False, load_cached_data=True)

    # Candidates for ensemble (Cycles 2 to N)
    # Cycle 1 is structural initialization, usually excluded from ensemble
    candidates = [f"best_cycle_{i}.pth" for i in range(2, Config.CYCLES + 1)]

    # Load available models
    models = {}
    for cp in candidates:
        path = os.path.join(Config.CHECKPOINT_DIR, cp)
        if os.path.exists(path):
            print(f"Loading checkpoint: {cp}")
            model = evaluator.load_model(path)
            if model is not None:
                models[cp] = model
        else:
            print(f"Checkpoint {cp} not found.")

    if not models:
        print("Error: No models trained/found. Exiting.")
        return

    # Run inference on validation set for all models
    print("Running inference on validation set...")

    candidate_preds = {cp: [] for cp in models}
    ground_truths = []
    val_ids = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(Config.DEVICE)
            depths = depths.to(Config.DEVICE)

            # Store Ground Truth (Cropped to original size)
            masks_cropped = masks[
                :,
                :,
                evaluator.crop_top : evaluator.crop_bottom,
                evaluator.crop_left : evaluator.crop_right,
            ]
            ground_truths.append(masks_cropped.cpu().numpy())
            val_ids.extend(ids)

            # Run inference for each candidate model
            for cp, model in models.items():
                # Predict with TTA
                probs = evaluator.predict_with_tta(model, images, depths)
                # Crop to original size
                probs_cropped = probs[
                    :,
                    :,
                    evaluator.crop_top : evaluator.crop_bottom,
                    evaluator.crop_left : evaluator.crop_right,
                ]
                candidate_preds[cp].append(probs_cropped.cpu().numpy())

    # Concatenate batches
    ground_truths = np.concatenate(ground_truths, axis=0).squeeze(1)  # (N, H, W)
    for cp in candidate_preds:
        candidate_preds[cp] = np.concatenate(candidate_preds[cp], axis=0).squeeze(
            1
        )  # (N, H, W)

    # Calculate Individual Scores and Apply Gating
    print("Calculating individual model scores...")
    scores = {}
    for cp in candidate_preds:
        score = do_kaggle_metric(candidate_preds[cp], ground_truths, threshold=0.5)
        scores[cp] = score
        print(f"  {cp}: {score:.6f}")

    # Gating Logic
    best_score = max(scores.values())
    threshold_score = best_score - Config.QUALITY_GATE_THRESHOLD
    selected_cps = [cp for cp, s in scores.items() if s >= threshold_score]

    print(f"Gating Threshold: {threshold_score:.6f} (Best: {best_score:.6f})")
    print(f"Selected Models for Ensemble: {selected_cps}")

    # Create Ensemble Prediction
    print("Ensembling selected models...")
    ensemble_preds = np.zeros_like(ground_truths, dtype=np.float32)
    for cp in selected_cps:
        ensemble_preds += candidate_preds[cp]
    ensemble_preds /= len(selected_cps)

    # Calculate Final Validation Metric
    final_metric = do_kaggle_metric(ensemble_preds, ground_truths, threshold=0.5)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate per-image mAP scores
    per_image_scores = []
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    # Binarize predictions and targets
    preds_bin = (ensemble_preds > 0.5).astype(np.uint8)
    targets_bin = (ground_truths > 0.5).astype(np.uint8)

    for i in range(len(preds_bin)):
        p = preds_bin[i]
        t = targets_bin[i]
        sum_p = p.sum()
        sum_t = t.sum()

        if sum_p == 0 and sum_t == 0:
            s = 1.0
        elif sum_p > 0 and sum_t == 0:
            s = 0.0
        elif sum_p == 0 and sum_t > 0:
            s = 0.0
        else:
            intersection = np.sum(p * t)
            union = sum_p + sum_t - intersection
            iou = intersection / union if union > 0 else 1.0
            matches = np.sum(iou > iou_thresholds)
            s = matches / 10.0
        per_image_scores.append(s)

    # Error is 1 - mAP
    errors = 1.0 - np.array(per_image_scores)

    # Map errors to metadata
    df_val = pd.read_csv(Config.VAL_CSV)
    error_map = {id_: err for id_, err in zip(val_ids, errors)}
    df_val["error"] = df_val["id"].map(error_map)

    # Calculate Correlations
    corr_depth, _ = pearsonr(df_val["z"], df_val["error"])
    corr_cov, _ = pearsonr(df_val["coverage"], df_val["error"])

    print(f"Correlation (Depth vs Error): {corr_depth:.10f}")
    print(f"Correlation (Coverage vs Error): {corr_cov:.10f}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.833:
        print("\n=== Generating Submission ===")
        # Use Evaluator to generate submission on test set
        evaluator.gated_ensemble()
    else:
        print(f"\nMetric {final_metric:.4f} <= 0.833. Submission skipped.")


if __name__ == "__main__":
    main()
