import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library import utils, train, model, data_loader, predict


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    utils.set_seed(Config.SEED)

    # Configure for Fast Baseline Execution
    # Reducing epochs to ensure completion within time limits while maintaining
    # sufficient convergence for the small dataset.
    Config.EPOCHS = 30
    Config.PATIENCE = 10

    # Setup directories
    Config.setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Training Loop & OOF Inference
    # -------------------------------------------------------------------------
    # We will collect Out-Of-Fold (OOF) predictions to compute the global CV metric
    oof_preds = []
    oof_targets = []
    oof_angles = []

    # Collectors for Failure Analysis features
    feat_b1_mean = []
    feat_b2_mean = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold in range(Config.N_FOLDS):
        print(f"\n" + "=" * 40)
        print(f"Processing Fold {fold}/{Config.N_FOLDS - 1}")
        print("=" * 40)

        # A. Train the model for this fold
        train.run_fold(fold)

        # B. Load the best checkpoint for inference on validation set
        print(f"Loading best model for Fold {fold}...")
        net = model.IAMSI_CNN().to(device)

        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"model_best_fold_{fold}.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"CRITICAL: Checkpoint {checkpoint_path} not found. Skipping fold.")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        # C. Get Validation Data
        # Note: We reload the loader to ensure we get the exact validation split
        _, val_loader = data_loader.get_train_val_loaders(fold, load_cached_data=True)

        # D. Run Inference
        print("Generating validation predictions...")
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Forward pass
                # Model returns averaged logits in eval mode
                logits = net(images, angles_gpu)
                probs = torch.sigmoid(logits)

                # Collect predictions and metadata
                oof_preds.extend(probs.cpu().numpy().flatten())
                oof_targets.extend(targets.numpy().flatten())
                oof_angles.extend(angles.numpy().flatten())

                # Feature Extraction for Failure Analysis
                # Images are (B, 3, 75, 75). Channel 0=HH, 1=HV
                imgs_np = images.cpu().numpy()
                # Compute simple stats
                feat_b1_mean.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
                feat_b2_mean.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))

    # -------------------------------------------------------------------------
    # 3. Global Validation Metric
    # -------------------------------------------------------------------------
    if not oof_preds:
        print("No predictions generated. Aborting.")
        return

    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)

    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred_clipped)

    print("\n" + "-" * 40)
    print(f"Final Validation Metric: {final_metric}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Calculate Log Loss contribution per sample
    # Loss = -(y*log(p) + (1-y)*log(1-p))
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "inc_angle": oof_angles,
            "b1_mean": feat_b1_mean,
            "b2_mean": feat_b2_mean,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error (Log Loss) and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    TARGET_THRESHOLD = 0.17174082291273365

    if final_metric < TARGET_THRESHOLD:
        print(f"\nMetric ({final_metric}) < Threshold ({TARGET_THRESHOLD}).")
        print("Proceeding with Test Set Inference and Submission Generation...")
        predict.generate_predictions(load_cached_data=True)
    else:
        print(f"\nMetric ({final_metric}) >= Threshold ({TARGET_THRESHOLD}).")
        print("Submission skipped.")


if __name__ == "__main__":
    main()
