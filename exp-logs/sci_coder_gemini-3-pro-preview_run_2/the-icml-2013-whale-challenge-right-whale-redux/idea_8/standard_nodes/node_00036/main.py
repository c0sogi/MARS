import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.models import WhaleEfficientNet, WhaleDenseNet
from library.train import train_model_instance
from library.inference import run_inference, predict


def perform_failure_analysis(model_a, model_b, val_loader, device):
    """
    Analyzes the ensemble's performance on the validation set.
    Computes correlations between error magnitude and input signal statistics.
    """
    print("\n--- Failure Analysis ---")
    model_a.eval()
    model_b.eval()

    all_targets = []
    all_probs_a = []
    all_probs_b = []

    # Feature lists
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target_np = target.numpy()

            # Extract features from spectrogram (Batch, Channel, Freq, Time)
            # We use the raw spectrogram values (already DB normalized in preprocessing)
            # Calculate stats per sample in the batch
            # Flatten spatial dims: (B, C, F, T) -> (B, F*T)
            flat_data = data.view(data.size(0), -1)

            feat_means.extend(flat_data.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_data.std(dim=1).cpu().numpy())
            feat_maxs.extend(flat_data.max(dim=1).values.cpu().numpy())

            # Get Predictions
            out_a = model_a(data)
            out_b = model_b(data)

            prob_a = torch.sigmoid(out_a).cpu().numpy().flatten()
            prob_b = torch.sigmoid(out_b).cpu().numpy().flatten()

            all_targets.extend(target_np)
            all_probs_a.extend(prob_a)
            all_probs_b.extend(prob_b)

    # Convert to arrays
    targets = np.array(all_targets)
    probs_a = np.array(all_probs_a)
    probs_b = np.array(all_probs_b)

    # Ensemble Prediction
    ensemble_probs = (probs_a + probs_b) / 2.0

    # Calculate Metric
    final_auc = roc_auc_score(targets, ensemble_probs)

    # Calculate Error Magnitude
    # Error = |Probability - Label|
    errors = np.abs(ensemble_probs - targets)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": np.array(feat_means),
            "spec_std": np.array(feat_stds),
            "spec_max": np.array(feat_maxs),
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")

    print(f"Correlation between Error Magnitude and Input Features:")
    print(correlations)

    return final_auc


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Define submission path as required by the task
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_file_path = os.path.join(submission_dir, "submission.csv")

    print(f"Device: {device}")
    print("Initializing Data Loaders...")

    # 2. Load Data
    # Using cached data to speed up loading if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Train Model A (EfficientNet-B0 Noisy Student)
    print("\n=== Training Model A: EfficientNet-B0 (Noisy Student) ===")
    model_a = train_model_instance(
        WhaleEfficientNet,
        "Model A",
        Config.MODEL_A_PATH,
        train_loader,
        val_loader,
        device,
    )

    # 4. Train Model B (DenseNet-121)
    print("\n=== Training Model B: DenseNet-121 ===")
    model_b = train_model_instance(
        WhaleDenseNet, "Model B", Config.MODEL_B_PATH, train_loader, val_loader, device
    )

    # 5. Validation and Failure Analysis
    # We pass the loaded models directly to avoid reloading from disk
    final_val_auc = perform_failure_analysis(model_a, model_b, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Submission Logic
    # Threshold defined in task description
    THRESHOLD = 0.9959177895986835

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Run inference using the ensemble
        # run_inference handles the loop, soft voting, and saving
        models = [model_a, model_b]
        run_inference(models, test_loader, device, output_path=submission_file_path)

    else:
        print(
            f"\nValidation metric ({final_val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
