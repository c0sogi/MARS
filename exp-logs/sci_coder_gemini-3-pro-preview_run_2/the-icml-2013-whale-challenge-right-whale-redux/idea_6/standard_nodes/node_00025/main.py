import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import roc_auc_score

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, get_test_loader
from library.model import WhaleEnsembleMember
from library.train import train_individual_model, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_failure_analysis(val_loader, models, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input signal statistics.
    """
    print("\n=== Failure Analysis ===")

    # Set models to eval
    for m in models:
        m.eval()

    all_targets = []
    all_preds = []

    # Features for correlation
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)  # (Batch,)

            # Ensemble Prediction
            batch_preds = []
            for m in models:
                logits = m(data)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy().flatten())

            # Average probabilities (Soft Voting)
            avg_preds = np.mean(batch_preds, axis=0)

            all_preds.extend(avg_preds)
            all_targets.extend(target.cpu().numpy())

            # Extract simple features from spectrograms for analysis
            # data shape: (B, 1, F, T)
            # We calculate stats per sample
            B = data.size(0)
            flat_data = data.view(B, -1)

            feat_means.extend(flat_data.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_data.std(dim=1).cpu().numpy())
            feat_maxs.extend(flat_data.max(dim=1).values.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": feat_means,
            "spec_std": feat_stds,
            "spec_max": feat_maxs,
        }
    )

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    for col in ["spec_mean", "spec_std", "spec_max"]:
        # Compute Pearson correlation
        corr = np.corrcoef(df_analysis["error"], df_analysis[col])[0, 1]
        print(f"  Error vs {col}: {corr:.16f}")

    return all_targets, all_preds


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    # Using load_cached_data=True to utilize preprocessed .npy files if available
    train_loader, val_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Training Loop
    trained_models = []

    for model_name in Config.MODEL_NAMES:
        # Train individual model
        # This function handles the training loop, validation monitoring, and saving the best checkpoint
        best_ckpt_path = train_individual_model(model_name, train_loader, val_loader)

        # Load Best Model for Inference
        print(f"Loading best weights for {model_name} from {best_ckpt_path}...")
        model = WhaleEnsembleMember(model_name, pretrained=False)
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
        trained_models.append(model)

    # 4. Ensemble Validation & Failure Analysis
    print("\nRunning Ensemble Validation...")
    targets, preds = run_failure_analysis(val_loader, trained_models, device)

    # Calculate Final Metric (AUC)
    try:
        final_metric = roc_auc_score(targets, preds)
    except ValueError:
        final_metric = 0.5

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # 5. Submission
    # Threshold defined in requirements
    threshold = 0.9959177895986835

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        test_loader, test_clips = get_test_loader(
            load_cached_data=True, debug=Config.DEBUG
        )

        ensemble_test_preds = []

        # Inference for each model in the ensemble
        for i, model in enumerate(trained_models):
            print(f"Inference with model {Config.MODEL_NAMES[i]}...")
            model_preds = inference(model, test_loader, device)
            ensemble_test_preds.append(model_preds)

        # Average predictions (Soft Voting)
        avg_test_preds = np.mean(ensemble_test_preds, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"clip": test_clips, "probability": avg_test_preds}
        )

        # Save to disk
        submission_df.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.OUTPUT_SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
