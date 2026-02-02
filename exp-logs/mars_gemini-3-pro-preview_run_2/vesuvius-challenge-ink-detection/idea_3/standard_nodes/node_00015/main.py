import os
import shutil
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

from library.config import WORKING_DIR, SUBMISSION_PATH, DEVICE, SEED
from library.train import train_model, set_seed
from library.models import build_model
from library.data import get_loaders
from library.utils import optimize_threshold
from library.inference import run_inference


def main():
    # 1. Setup
    set_seed(SEED)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    final_submission_path = os.path.join(submission_dir, "submission.csv")

    baseline_threshold = 0.4738558828830719

    print("=== Starting Fast Baseline Run ===")

    # 2. Train Model
    # We limit samples and epochs to ensure execution within the time limit.
    # train_model will save 'best_model.pth' in WORKING_DIR and generate 'submission.csv'
    # in the root if the validation score exceeds the baseline.
    print("Training model...")
    _ = train_model(
        max_train_samples=2000,
        num_epochs=5,
        patience=3,
        baseline_score=baseline_threshold,
    )

    # 3. Validation & Failure Analysis
    print("\n=== Performing Validation & Failure Analysis ===")

    # Load the best model
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("No best model found. Training might have failed to converge.")
        return

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Get validation loader
    _, val_loader, _ = get_loaders(batch_size=8, num_workers=2)

    all_preds = []
    all_targets = []

    # buffers for failure analysis
    fa_errors = []
    fa_ch_max = []
    fa_ch_mean = []
    fa_ch_std = []

    # We analyze a subset of pixels for failure analysis to avoid OOM,
    # but we use all for the metric calculation.
    analysis_batches = 20

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Inference (no grad)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Collect data for metric
            preds_np = probs.cpu().numpy().flatten()
            targets_np = labels.cpu().numpy().flatten()

            all_preds.append(preds_np)
            all_targets.append(targets_np)

            # Collect data for failure analysis (subset)
            if i < analysis_batches:
                # Error = |Prob - Target|
                error = np.abs(preds_np - targets_np)

                # Input features
                imgs_np = images.cpu().numpy()
                # Channel 0: Max, 1: Mean, 2: Std
                ch_max = imgs_np[:, 0, :, :].flatten()
                ch_mean = imgs_np[:, 1, :, :].flatten()
                ch_std = imgs_np[:, 2, :, :].flatten()

                fa_errors.append(error)
                fa_ch_max.append(ch_max)
                fa_ch_mean.append(ch_mean)
                fa_ch_std.append(ch_std)

    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Final Metric
    # We re-optimize threshold on the full validation set to get the precise metric
    best_thresh, final_metric = optimize_threshold(all_preds, all_targets, beta=0.5)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    flat_errors = np.concatenate(fa_errors)
    flat_max = np.concatenate(fa_ch_max)
    flat_mean = np.concatenate(fa_ch_mean)
    flat_std = np.concatenate(fa_ch_std)

    corr_max, _ = pearsonr(flat_errors, flat_max)
    corr_mean, _ = pearsonr(flat_errors, flat_mean)
    corr_std, _ = pearsonr(flat_errors, flat_std)

    print("-" * 30)
    print("Failure Analysis: Correlation between Error and Input Features")
    print(f"Max Intensity (MIP): {corr_max:.6f}")
    print(f"Mean Intensity:      {corr_mean:.6f}")
    print(f"Std Deviation:       {corr_std:.6f}")
    print("-" * 30)

    # 4. Submission Logic
    # train_model generates submission at SUBMISSION_PATH (root/submission.csv)
    # if its internal check passed.

    submission_generated = False

    # Check if train_model already generated it
    if os.path.exists(SUBMISSION_PATH):
        print(
            f"Moving generated submission from {SUBMISSION_PATH} to {final_submission_path}"
        )
        shutil.move(SUBMISSION_PATH, final_submission_path)
        submission_generated = True

    # If not generated, but our re-calculated metric is high enough, generate it now.
    if not submission_generated and final_metric > baseline_threshold:
        print(
            f"Metric {final_metric} exceeds baseline. Generating submission explicitly..."
        )
        run_inference(
            model_path=model_path,
            threshold=best_thresh,
            submission_output=final_submission_path,
        )
        submission_generated = True

    if submission_generated:
        print(f"Submission successfully saved to {final_submission_path}")
    else:
        print(
            f"Metric {final_metric} did not exceed baseline {baseline_threshold}. No submission generated."
        )


if __name__ == "__main__":
    main()
