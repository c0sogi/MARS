import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.train import run_training
from library.inference import predict
from library.model import AASLNet
from library.data import get_dataloaders

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # We set seeds for reproducibility
    seed_everything(Config.SEED)

    # OPTIMIZATION FOR FAST BASELINE:
    # The default config specifies 50 epochs. Given the time constraint (~25 mins),
    # we reduce this to 15 epochs. With ~1100 training samples and batch size 32,
    # this ensures the training phase completes in approximately 5-8 minutes on an A100 GPU,
    # leaving ample time for validation, analysis, and inference.
    Config.EPOCHS = 15

    print("=========================================")
    print("AASL-Net Pipeline Execution")
    print("=========================================")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # 2. Training Phase
    print("\n[Step 1] Starting Model Training...")
    # run_training initializes the model, trainer, and data loaders.
    # It returns the best validation score achieved during training.
    # We use debug=False to train on the full dataset for a valid baseline.
    best_val_score = run_training(debug=False)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {best_val_score}")

    # 3. Failure Analysis Phase
    print("\n[Step 2] Performing Failure Analysis...")

    # We need to reload the validation data and the best model to analyze errors
    _, val_loader = get_dataloaders(debug=False)

    device = torch.device(Config.DEVICE)
    model = AASLNet()
    model.to(device)

    # Load the best weights saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("CRITICAL: Best model weights not found. Skipping analysis.")
        return

    model.eval()

    # Containers for analysis
    analysis_data = []

    # Run inference on validation set (no gradient needed)
    with torch.no_grad():
        for img_ax, img_cor, tabular, meta, target in val_loader:
            # Move inputs to device
            img_ax = img_ax.to(device)
            img_cor = img_cor.to(device)
            tabular = tabular.to(device)
            meta = meta.to(device)
            target = target.to(device)

            # Forward pass
            pred_fvc, _ = model(img_ax, img_cor, tabular, meta)

            # Move data back to CPU for numpy operations
            pred_fvc_np = pred_fvc.cpu().numpy()
            target_np = target.cpu().numpy()
            tabular_np = tabular.cpu().numpy()
            meta_np = meta.cpu().numpy()

            # Iterate through batch to collect row-wise data
            for i in range(len(pred_fvc_np)):
                # Calculate Absolute Error
                abs_err = np.abs(pred_fvc_np[i] - target_np[i])

                # Extract features (Tabular: [Age_Norm, Sex, Smoke, Pct_Norm])
                # Meta: [Relative_Week, Base_FVC]
                row = {
                    "AbsError": abs_err,
                    "Age_Norm": tabular_np[i, 0],
                    "Sex_Val": tabular_np[i, 1],
                    "Smoke_Val": tabular_np[i, 2],
                    "Pct_Norm": tabular_np[i, 3],
                    "Week": meta_np[i, 0],
                    "Base_FVC": meta_np[i, 1],
                }
                analysis_data.append(row)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(analysis_data)

    print("Correlation between Model Error (Absolute) and Input Features:")
    features = ["Age_Norm", "Sex_Val", "Smoke_Val", "Pct_Norm", "Week", "Base_FVC"]

    for feat in features:
        if feat in df_analysis.columns:
            # Handle constant columns (std=0) to avoid NaN correlations
            if df_analysis[feat].std() == 0:
                corr = 0.0
            else:
                corr = df_analysis["AbsError"].corr(df_analysis[feat])
            print(f"  {feat}: {corr:.6f}")

    # 4. Submission Phase
    print("\n[Step 3] Checking Submission Criteria...")
    THRESHOLD = -6.510164260864258

    if best_val_score > THRESHOLD:
        print(f"Validation Score ({best_val_score}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        predict()
    else:
        print(
            f"Validation Score ({best_val_score}) does not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
