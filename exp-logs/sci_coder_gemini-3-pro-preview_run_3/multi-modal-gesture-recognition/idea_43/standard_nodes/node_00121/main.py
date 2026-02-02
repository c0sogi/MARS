import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.nn.functional as F

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.data_loader import get_dataloaders
from library.utils import decode_predictions, compute_levenshtein_distance


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)


def main():
    print("=== Physically-Aligned Moderate-Capacity Network (PAM-CN) Execution ===")

    # 1. Initialize Trainer
    trainer = Trainer()

    # 2. Training
    # We limit epochs to 30 for a fast baseline execution as requested.
    # The dataset is small, so this should be sufficient for convergence or early stopping.
    print("\n[Step 1/4] Starting Training...")
    trainer.fit(epochs=30, patience=8)

    # 3. Validation & Metrics
    print("\n[Step 2/4] Validating Best Model...")
    trainer.load_best_model()
    trainer.model.eval()

    # Get loaders (re-using cache)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    total_dist = 0
    total_gestures = 0

    # Storage for Failure Analysis
    analysis_records = []

    device = trainer.device

    with torch.no_grad():
        for features, labels, sample_id in val_loader:
            features = features.to(device)
            # labels shape: (1, T)
            labels_np = labels.squeeze(0).cpu().numpy()

            # Forward Pass
            # PAM-CN returns (logits1, logits2, logits3)
            _, _, logits3 = trainer.model(features)

            # Decode Predictions
            probs3 = F.softmax(logits3, dim=2)
            # argmax over classes (dim 2) -> (1, T)
            preds_frame = torch.argmax(probs3, dim=2).squeeze(0).cpu().numpy()

            # Convert to gesture lists
            pred_gestures = decode_predictions(preds_frame)
            true_gestures = decode_predictions(labels_np)

            # Compute Metric for this sample
            dist = compute_levenshtein_distance(pred_gestures, true_gestures)

            total_dist += dist
            total_gestures += len(true_gestures)

            # Collect data for failure analysis
            seq_len = features.shape[1]
            num_true = len(true_gestures)

            # Calculate simple input statistics
            # features: (1, T, 193)
            feat_np = features.squeeze(0).cpu().numpy()
            # Mean absolute magnitude of input features
            mean_signal = np.mean(np.abs(feat_np))

            analysis_records.append(
                {
                    "sample_id": sample_id[0],
                    "error": dist,
                    "seq_len": seq_len,
                    "num_gestures": num_true,
                    "mean_signal": mean_signal,
                }
            )

    # Compute Final Metric
    if total_gestures > 0:
        final_metric = total_dist / total_gestures
    else:
        final_metric = float("inf")

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n[Step 3/4] Performing Failure Analysis...")
    if len(analysis_records) > 0:
        df_analysis = pd.DataFrame(analysis_records)

        # Calculate correlations with Error
        # We look at absolute error magnitude
        target_col = "error"
        feature_cols = ["seq_len", "num_gestures", "mean_signal"]

        print("Correlation between Error Magnitude and Input Features:")
        for col in feature_cols:
            if df_analysis[col].std() > 1e-9:  # Avoid division by zero
                corr, _ = pearsonr(df_analysis[target_col], df_analysis[col])
                print(f"  Correlation (Error vs {col}): {corr:.4f}")
            else:
                print(f"  Correlation (Error vs {col}): Undefined (Constant feature)")

        # Optional: Print worst performing samples
        df_analysis["error_rate"] = df_analysis.apply(
            lambda row: (
                row["error"] / row["num_gestures"]
                if row["num_gestures"] > 0
                else row["error"]
            ),
            axis=1,
        )
        worst_samples = df_analysis.sort_values("error", ascending=False).head(3)
        print("\nTop 3 Samples with Highest Absolute Error:")
        for _, row in worst_samples.iterrows():
            print(
                f"  ID: {row['sample_id']}, Error: {row['error']}, True Gestures: {row['num_gestures']}"
            )

    # 5. Submission
    print("\n[Step 4/4] Checking Submission Criteria...")
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.5f}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_test_set()
    else:
        print(
            f"Metric ({final_metric:.5f}) is NOT lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
