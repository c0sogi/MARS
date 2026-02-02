import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
import gc

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import (
    PATHS,
    TRAINING_PARAMS,
    SPECIALIST_SETTINGS,
    MODEL_PARAMS,
    SLAB_PARAMS,
    DEVICE,
    SEED,
)
from library import train_engine
from library import inference_engine
from library.model import get_model
from library.data_utils import get_fragment_3ch_slab


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    set_seed(SEED)

    # --- 1. Training Phase ---
    # Limiting epochs to 10 to ensure execution within time limits.
    # The dataset is small (412 patches), so 10 epochs is sufficient for a baseline.
    TRAINING_PARAMS["epochs"] = 10

    modes = ["High", "Mid", "Low"]
    print("Starting Training Phase...")

    for mode in modes:
        # Train each specialist model.
        # load_cached_data=True allows skipping slab generation if files exist in ./working
        train_engine.run_specialist_training(mode, load_cached_data=True)

        # Cleanup to free GPU memory for the next run
        gc.collect()
        torch.cuda.empty_cache()

    # --- 2. Validation Phase (Ensemble Evaluation) ---
    print("\nStarting Ensemble Validation...")

    # Load Validation Metadata
    if not os.path.exists(PATHS.VAL_METADATA):
        print("Validation metadata not found.")
        return
    val_df = pd.read_csv(PATHS.VAL_METADATA)

    # Load Trained Models
    models = {}
    for mode in modes:
        model_path = os.path.join(PATHS.WORKING_DIR, f"model_{mode}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model {mode} not found. Skipping validation for this mode."
            )
            continue

        m = get_model(MODEL_PARAMS)
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        models[mode] = m

    if not models:
        print("No models available for validation.")
        return

    # Preload Slabs for Validation Fragments
    # Validation patches are subsets of training fragments. We load the full slabs once.
    val_frag_ids = val_df["fragment_id"].unique()
    slabs_cache = {mode: {} for mode in modes}

    print("Preloading validation slabs...")
    for fid in val_frag_ids:
        fid_str = str(fid)
        for mode in modes:
            if mode not in models:
                continue
            settings = SPECIALIST_SETTINGS[mode]
            # Validation data uses 'train' split source files
            slab = get_fragment_3ch_slab(
                fragment_id=fid_str,
                split="train",
                z_start=settings["z_start"],
                z_end=settings["z_end"],
                slab_params=SLAB_PARAMS,
                load_cached_data=True,
            )
            slabs_cache[mode][fid_str] = slab

    # Load Ground Truth Labels
    labels_cache = {}
    for fid in val_frag_ids:
        fid_str = str(fid)
        label_path = os.path.join(PATHS.TRAIN_FRAGMENTS, fid_str, "inklabels.png")
        if os.path.exists(label_path):
            lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # Binarize
            labels_cache[fid_str] = (lbl > 0).astype(np.uint8)

    # Metrics Accumulators
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Failure Analysis Data
    error_magnitudes = []
    input_intensities = []

    print("Evaluating validation patches...")
    with torch.no_grad():
        for idx, row in val_df.iterrows():
            fid = str(row["fragment_id"])
            x, y = row["x"], row["y"]
            w, h = row["width"], row["height"]

            patch_probs = []
            patch_intensities = []

            # Inference per specialist
            for mode in modes:
                if mode not in models:
                    continue

                # Extract crop from preloaded slab
                full_slab = slabs_cache[mode][fid]
                patch = full_slab[y : y + h, x : x + w, :]  # (H, W, 3)

                # Store intensity for failure analysis
                patch_intensities.append(np.mean(patch))

                # Prepare tensor: (H, W, 3) -> (1, 3, H, W)
                input_tensor = (
                    torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
                )

                # Predict
                logits = models[mode](input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()  # (H, W)
                patch_probs.append(probs)

            if not patch_probs:
                continue

            # Ensemble Fusion: Max Probability Projection
            final_prob = np.max(np.stack(patch_probs), axis=0)
            pred_mask = (final_prob > 0.5).astype(np.uint8)

            # Ground Truth
            full_label = labels_cache[fid]
            target_mask = full_label[y : y + h, x : x + w]

            # Update Global Counts
            p_flat = pred_mask.flatten()
            t_flat = target_mask.flatten()

            tp = np.sum(p_flat * t_flat)
            fp = np.sum(p_flat * (1 - t_flat))
            fn = np.sum((1 - p_flat) * t_flat)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            # Failure Analysis: Mean Absolute Error vs Intensity
            mae = np.mean(np.abs(final_prob - target_mask))
            error_magnitudes.append(mae)
            input_intensities.append(np.mean(patch_intensities))

    # Compute Final F0.5 Score
    beta = 0.5
    epsilon = 1e-7
    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)

    f05_score = (
        (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + epsilon)
    )

    print(f"Final Validation Metric: {f05_score}")

    # --- 3. Failure Analysis ---
    print("\nFailure Analysis:")
    if len(error_magnitudes) > 1:
        corr, _ = pearsonr(error_magnitudes, input_intensities)
        print(f"Correlation between Error Magnitude and Input Intensity: {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # --- 4. Submission ---
    # Threshold check
    THRESHOLD_SCORE = 0.597622633
    if f05_score > THRESHOLD_SCORE:
        print(f"\nValidation score exceeds {THRESHOLD_SCORE}. Generating submission...")

        # Clear memory before heavy inference
        del slabs_cache, models, labels_cache
        gc.collect()
        torch.cuda.empty_cache()

        # Run Inference Engine
        engine = inference_engine.InferenceEngine()
        engine.generate_submission()
    else:
        print(
            f"\nValidation score {f05_score} did not exceed {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
