import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import get_model
from library.transforms import AudioTransforms
from library.utils import set_seed, map_fine_to_coarse
from library.trainer import run_training


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training
    # We run for 8 epochs to ensure a fast baseline execution that fits within the time limit.
    # The efficientnet backbone converges quickly with pre-trained weights.
    print("Starting training pipeline...")
    best_model_path = run_training(epochs=8, load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("Running validation inference...")

    # Load DataLoaders (using cached data for speed)
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Load the best model
    model = get_model(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Initialize Transforms (Inference Mode)
    transforms = AudioTransforms(device=device)

    all_preds_fine = []
    all_labels_fine = []

    # Validation Inference Loop
    with torch.no_grad():
        for waveforms, labels, _ in val_loader:
            waveforms = waveforms.to(device)
            # labels are not needed for transforms in eval mode, but passed for consistency if needed
            features = transforms(waveforms, labels=None, train=False)

            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            all_preds_fine.extend(predicted.cpu().numpy())
            all_labels_fine.extend(labels.cpu().numpy())

    # Map Fine-Grained Predictions to Competition Targets
    # Convert indices to string labels first
    pred_labels_str = [Config.get_label_from_index(idx) for idx in all_preds_fine]
    true_labels_str = [Config.get_label_from_index(idx) for idx in all_labels_fine]

    # Map to the 12 target classes (yes, no, ..., unknown, silence)
    mapped_preds = map_fine_to_coarse(pred_labels_str)
    mapped_true = map_fine_to_coarse(true_labels_str)

    # Calculate Final Metric
    final_metric = accuracy_score(mapped_true, mapped_preds)

    # Print Metric in required format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")

    # Create binary error vector (1 = Error, 0 = Correct)
    errors = np.array([1 if p != t else 0 for p, t in zip(mapped_preds, mapped_true)])

    # Input Feature for correlation: Fine-Grained Label Index
    # This helps identify if specific source words are systematically harder
    label_feature = np.array(all_labels_fine)

    if len(errors) > 0 and np.std(errors) > 0:
        corr_label = np.corrcoef(errors, label_feature)[0, 1]
        print(
            f"Correlation between Error Magnitude and Input Label Index: {corr_label}"
        )
    else:
        print("Failure analysis skipped: No errors or insufficient variance.")

    # 5. Submission Generation
    THRESHOLD = 0.9872909698996656

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_preds_fine = []
        test_fnames = []

        # Test Inference Loop
        with torch.no_grad():
            for waveforms, _, fnames in test_loader:
                waveforms = waveforms.to(device)

                features = transforms(waveforms, labels=None, train=False)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)

                test_preds_fine.extend(predicted.cpu().numpy())
                test_fnames.extend(fnames)

        # Map Test Predictions
        pred_labels_str_test = [
            Config.get_label_from_index(idx) for idx in test_preds_fine
        ]
        mapped_preds_test = map_fine_to_coarse(pred_labels_str_test)

        # Create Submission File
        submission_df = pd.DataFrame({"fname": test_fnames, "label": mapped_preds_test})

        # Save to ./submission/submission.csv
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
