import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import AudioEfficientNet
from library.train import run_training


def evaluate_model(model, data_loader, device):
    """
    Runs inference on a dataloader and returns predictions, targets, and input stats.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    # For failure analysis
    input_means = []
    input_stds = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Calculate simple input stats for failure analysis
            # inputs shape: (B, 1, F, T)
            # We calculate mean/std per sample
            B = inputs.size(0)
            flat_inputs = inputs.view(B, -1)
            input_means.extend(flat_inputs.mean(dim=1).cpu().numpy())
            input_stds.extend(flat_inputs.std(dim=1).cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
        np.array(input_means),
        np.array(input_stds),
    )


def perform_failure_analysis(preds, targets, probs, input_means, input_stds):
    """
    Correlates error magnitude with input features.
    """
    # Calculate Error Magnitude: 1.0 - probability of the correct class
    # For correct predictions, this is low. For incorrect, it is high.
    # We use the probability assigned to the TRUE class.

    # Get probability of true class for each sample
    true_class_probs = probs[np.arange(len(targets)), targets]
    error_magnitude = 1.0 - true_class_probs

    df_analysis = pd.DataFrame(
        {
            "error_magnitude": error_magnitude,
            "input_mean": input_means,
            "input_std": input_stds,
            "is_correct": (preds == targets),
        }
    )

    print("\n=== Failure Analysis ===")
    print(f"Total Samples: {len(df_analysis)}")
    print(
        f"Misclassified Samples: {len(df_analysis) - df_analysis['is_correct'].sum()}"
    )

    # Correlation matrix
    correlations = df_analysis[["error_magnitude", "input_mean", "input_std"]].corr()[
        "error_magnitude"
    ]
    print("\nCorrelation with Error Magnitude:")
    print(correlations.drop("error_magnitude"))  # Don't print self-correlation

    return df_analysis


def generate_submission(model, test_loader, device):
    """
    Generates submission file for test set.
    """
    print("\nGenerating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())

    # Load test metadata to get filenames
    df_test = pd.read_csv(Config.TEST_CSV)

    # Map IDs to Labels
    predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

    # Create submission DataFrame
    submission = pd.DataFrame({"fname": df_test["fname"], "label": predicted_labels})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Train Model
    # We use the full dataset to ensure high accuracy.
    # A100 is fast enough for 20 epochs on this dataset size.
    print("Starting training process...")
    best_train_val_acc = run_training(
        debug=False, load_cached_data=True, epochs=Config.NUM_EPOCHS, patience=5
    )

    # 3. Load Best Model for Validation
    print("\nLoading best model for evaluation...")
    model = AudioEfficientNet(num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    # Get loaders
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True, debug=False)

    # 4. Final Validation Metric
    print("Running validation inference...")
    val_preds, val_targets, val_probs, val_means, val_stds = evaluate_model(
        model, val_loader, device
    )

    final_acc = np.mean(val_preds == val_targets)
    print(f"Final Validation Metric: {final_acc}")

    # 5. Failure Analysis
    perform_failure_analysis(val_preds, val_targets, val_probs, val_means, val_stds)

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.9843632410736683

    if final_acc > THRESHOLD:
        print(
            f"\nValidation metric ({final_acc}) > Threshold ({THRESHOLD}). Proceeding to submission."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({final_acc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
