import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import SEED, EPOCHS, NUM_FOLDS, WORKING_DIR, DEVICE
from library.utils import seed_everything
from library.model import get_data, generate_submission, ShallowUNet
from library.train_engine import run_fold


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Load Data
    # Use cached data if available for speed
    (train_ids, train_noisy, train_clean), (test_ids, test_noisy) = get_data(
        load_cached_data=True
    )

    # 3. Training Loop (5-Fold CV)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []
    failure_analysis_data = []

    print(
        f"Starting {NUM_FOLDS}-Fold Cross-Validation with {EPOCHS} epochs per fold..."
    )

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_noisy)):
        # Prepare data for this fold
        t_imgs = train_noisy[train_idx]
        t_masks = train_clean[train_idx]
        v_imgs = train_noisy[val_idx]
        v_masks = train_clean[val_idx]
        v_ids = train_ids[val_idx]

        # Train the fold
        # run_fold trains the model, saves the best checkpoint, and returns the best val RMSE
        best_rmse = run_fold(fold, t_imgs, t_masks, v_imgs, v_masks, epochs=EPOCHS)
        fold_scores.append(best_rmse)

        # --- Failure Analysis Data Collection ---
        # Reload best model for this fold to generate predictions for analysis
        model_path = os.path.join(WORKING_DIR, f"model_fold_{fold}.pth")
        model = ShallowUNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            for i, img in enumerate(v_imgs):
                # Preprocess: Normalize and convert to tensor
                img_norm = img.astype(np.float32) / 255.0
                tensor_img = (
                    torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
                )

                # Handle padding for inference (ShallowUNet requires dimensions divisible by 4)
                h, w = tensor_img.shape[2], tensor_img.shape[3]
                pad_h = (4 - h % 4) % 4
                pad_w = (4 - w % 4) % 4
                if pad_h > 0 or pad_w > 0:
                    tensor_img = torch.nn.functional.pad(
                        tensor_img, (0, pad_w, 0, pad_h)
                    )

                # Predict
                pred = model(tensor_img)

                # Crop back to original size
                if pad_h > 0 or pad_w > 0:
                    pred = pred[:, :, :h, :w]

                pred_np = pred.squeeze().cpu().numpy()
                target_np = v_masks[i].astype(np.float32) / 255.0

                # Compute RMSE for this specific image
                img_rmse = np.sqrt(np.mean((target_np - pred_np) ** 2))

                # Compute Image stats (Mean and Std of input noisy image)
                img_mean = np.mean(img_norm)
                img_std = np.std(img_norm)

                failure_analysis_data.append(
                    {
                        "id": v_ids[i],
                        "rmse": img_rmse,
                        "mean_intensity": img_mean,
                        "std_intensity": img_std,
                    }
                )

    # 4. Validation Assessment
    final_metric = np.mean(fold_scores)
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
