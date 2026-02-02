import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import from provided libraries
from library.config import Config
from library.trainer import Trainer
from library.model import get_model
from library.dataset import get_loaders
from library.inference import generate_submission


def main():
    # Set seed for reproducibility
    Config.set_seed(Config.SEED)

    print("==========================================")
    print("Starting Runfile Execution")
    print("==========================================")

    # ---------------------------------------------------------
    # 1. Training
    # ---------------------------------------------------------
    # We use 10 epochs to allow convergence with Focal Loss.
    print("\n[Step 1] Initializing Training...")
    trainer = Trainer(debug=False, epochs=10)

    print("Starting training loop...")
    trainer.fit()

    # ---------------------------------------------------------
    # 2. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 2] Performing Validation and Failure Analysis...")

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    best_model_path = Config.MODEL_SAVE_PATH

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    print(f"Loading best model from {best_model_path}...")
    model = get_model(device=device, weights_path=best_model_path)
    model.eval()

    # Get validation loader
    # We use the standard validation loader provided by the library
    _, val_loader, _ = get_loaders(debug=False)

    all_preds = []
    all_labels = []
    error_analysis_data = []

    print("Running validation inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Collect data for failure analysis
            # We compute stats on the CPU to avoid complex GPU indexing for logging
            imgs_np = images.cpu().numpy()  # Shape: (B, 3, H, W)
            lbls_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Iterate through batch to collect per-sample stats
            for i in range(len(imgs_np)):
                img = imgs_np[i]
                true_label = lbls_np[i]
                pred_label = preds_np[i]

                # Check if prediction is wrong (Binary Error)
                is_error = 1 if true_label != pred_label else 0

                # Calculate simple image statistics (per channel)
                # img is (3, H, W), axis=(1, 2) computes mean/std over H and W
                mean_ch = np.mean(img, axis=(1, 2))  # (3,)
                std_ch = np.std(img, axis=(1, 2))  # (3,)

                error_analysis_data.append(
                    {
                        "is_error": is_error,
                        "mean_r": mean_ch[0],
                        "mean_g": mean_ch[1],
                        "mean_b": mean_ch[2],
                        "std_r": std_ch[0],
                        "std_g": std_ch[1],
                        "std_b": std_ch[2],
                        "true_class": true_label,
                    }
                )

    # Calculate and Print Final Metric
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis: Correlation
    if error_analysis_data:
        df_error = pd.DataFrame(error_analysis_data)

        # Calculate correlation between Error and Input Features (Image Stats)
        features = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
        correlations = (
            df_error[features + ["is_error"]].corr()["is_error"].drop("is_error")
        )

        print("\nFailure Analysis - Correlation between Error and Input Features:")
        print(correlations)

        # Additional Insight: Error rate by class
        print("\nTop 5 Classes with Highest Error Rate:")
        class_error = (
            df_error.groupby("true_class")["is_error"]
            .mean()
            .sort_values(ascending=False)
        )
        print(class_error.head(5))

    # ---------------------------------------------------------
    # 3. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.3496646080071538
    if val_f1 > THRESHOLD:
        print("\n[Step 3] Generating Submission...")
        generate_submission(weights_path=best_model_path, debug=False)
    else:
        print(
            f"\n[Step 3] Validation F1 ({val_f1}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )

    print("\nRunfile execution completed successfully.")


if __name__ == "__main__":
    main()
