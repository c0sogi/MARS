import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import SEED, EPOCHS, NUM_FOLDS, WORKING_DIR, DEVICE
from library.utils import seed_everything
from library.model import get_data, generate_submission, UNet, predict_tta
from library.train_engine import run_fold


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Load Data
    # Load separate train and val sets (Cite Lesson 00016: Use official split)
    (
        (train_ids, train_noisy, train_clean),
        (val_ids, val_noisy, val_clean),
        (test_ids, test_noisy),
    ) = get_data(load_cached_data=True)

    # 3. Training Loop (Single Split)
    # Cite Lesson 00012: Prioritize single converged model over ensemble
    print(f"Starting Training with {EPOCHS} epochs...")

    # Train on the official split (Fold 0)
    _ = run_fold(0, train_noisy, train_clean, val_noisy, val_clean, epochs=EPOCHS)

    # 4. Validation Assessment with TTA
    # Cite Lesson 00025: Ensure parity by using TTA for final validation metric
    print("Computing Final Validation Metric with TTA...")

    model_path = os.path.join(WORKING_DIR, "model_fold_0.pth")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    failure_analysis_data = []
    total_rmse = 0.0

    with torch.no_grad():
        for i, img in enumerate(val_noisy):
            # Preprocess
            img_norm = img.astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Padding
            h, w = tensor_img.shape[2], tensor_img.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h > 0 or pad_w > 0:
                tensor_img = torch.nn.functional.pad(tensor_img, (0, pad_w, 0, pad_h))

            # Predict with TTA
            pred = predict_tta(model, tensor_img)

            # Crop back
            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :h, :w]

            pred_np = pred.squeeze().cpu().numpy()
            target_np = val_clean[i].astype(np.float32) / 255.0

            # Compute RMSE
            img_rmse = np.sqrt(np.mean((target_np - pred_np) ** 2))
            total_rmse += img_rmse

            # Failure Analysis Stats
            failure_analysis_data.append(
                {
                    "id": val_ids[i],
                    "rmse": img_rmse,
                    "mean_intensity": np.mean(img_norm),
                    "std_intensity": np.std(img_norm),
                }
            )

    final_metric = total_rmse / len(val_noisy)
    # Print full precision as requested
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis Report
    df_analysis = pd.DataFrame(failure_analysis_data)

    if not df_analysis.empty:
        # Calculate correlations
        corr_mean, _ = pearsonr(df_analysis["rmse"], df_analysis["mean_intensity"])
        corr_std, _ = pearsonr(df_analysis["rmse"], df_analysis["std_intensity"])

        print("-" * 30)
        print("Failure Analysis")
        print("-" * 30)
        print(f"Correlation (RMSE vs Mean Intensity): {corr_mean:.4f}")
        print(f"Correlation (RMSE vs Std Intensity): {corr_std:.4f}")
        print("-" * 30)

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.012221260240721992

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        # generate_submission handles loading the ensemble, TTA, and saving the CSV
        generate_submission(test_ids, test_noisy)
    else:
        print(
            f"Validation metric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
