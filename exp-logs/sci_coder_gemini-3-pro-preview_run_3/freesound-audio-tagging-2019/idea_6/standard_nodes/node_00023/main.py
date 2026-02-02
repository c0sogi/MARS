import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr

from library.configuration import Config
from library.utilities import set_seed, calculate_lrap
from library.network import ConvNeXtAudio
from library.data_loader import get_dataloaders, get_test_dataloader
from library.trainer import run_training


def main():
    # 1. Configuration Adjustments for Fast Baseline
    # Adjust epochs to ensure completion within 2 hours while maintaining performance
    Config.EPOCHS = 15

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_file_path = os.path.join(submission_dir, "submission.csv")

    print("Starting Runfile Execution...")
    print(f"Configuration: {Config.PROJECT_NAME}, Epochs: {Config.EPOCHS}")

    # 2. Run Training
    # This will train the model, save the best checkpoint, and return the best score.
    # We use load_cached_data=True to utilize pre-processed numpy arrays if available.
    best_val_score = run_training(config=Config, load_cached_data=True)

    # 3. Validation Inference & Metric Confirmation
    print("\n--- Starting Final Validation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)
    model = ConvNeXtAudio(config=Config)

    # Load the best model weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get Validation Loader
    _, val_loader = get_dataloaders(Config, load_cached_data=True)

    # Run Inference on Validation Set
    all_probs = []
    all_targets = []

    # Criterion for failure analysis (BCE per sample)
    criterion_reduction_none = nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Calculate loss per sample (mean over classes)
            loss_per_sample = criterion_reduction_none(logits, labels).mean(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_losses.append(loss_per_sample.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_losses = np.concatenate(all_losses)

    # Calculate Final Metric
    final_metric = calculate_lrap(all_targets, all_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")

    # Load Validation Metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Feature 1: Number of Labels (Complexity)
    # Handle potential NaNs or empty strings if any (though preprocessing should have cleaned this)
    val_df["num_labels"] = val_df["labels"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )

    # Ensure alignment: The loader maps indices 1-to-1 with the CSV if shuffle=False (which it is for val)
    # However, get_data might have filtered debug subset.
    # Since we are not in DEBUG mode (Config.DEBUG=False), lengths should match.
    if len(val_df) != len(all_losses):
        print(
            f"Warning: Metadata length ({len(val_df)}) matches predictions ({len(all_losses)})? "
            f"{len(val_df) == len(all_losses)}"
        )
        # Truncate to match if necessary (e.g. drop_last issues, though val loader drop_last=False)
        min_len = min(len(val_df), len(all_losses))
        val_df = val_df.iloc[:min_len]
        all_losses = all_losses[:min_len]

    # Correlation: Error Magnitude (Loss) vs Number of Labels
    corr_num_labels, _ = pearsonr(all_losses, val_df["num_labels"])
    print(f"Correlation (Error Magnitude vs Num Labels): {corr_num_labels:.4f}")

    # 5. Submission Generation
    threshold = 0.7117108825122853

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )

        test_loader = get_test_dataloader(Config, load_cached_data=True)

        test_probs = []
        test_fnames = []  # We need to ensure order matches, though loader preserves it.

        # We can retrieve fnames from the dataset inside the loader
        # But it's safer to rely on the loader iteration if we trust the order
        # The get_test_dataloader loads data using get_data('test'), which returns sorted fnames usually

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.cpu().numpy())

        test_probs = np.concatenate(test_probs)

        # Load Sample Submission to get correct column order and fnames
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # The test_loader loads files based on metadata/test.csv which is derived from sample_submission.
        # So the order should be identical.
        # Let's verify shape
        if len(test_probs) != len(sample_sub):
            print(
                f"Error: Prediction count {len(test_probs)} != Sample Submission count {len(sample_sub)}"
            )
        else:
            # Create Submission DataFrame
            submission_df = pd.DataFrame(test_probs, columns=sample_sub.columns[1:])
            submission_df.insert(0, "fname", sample_sub["fname"])

            # Save
            submission_df.to_csv(submission_file_path, index=False)
            print(f"Submission saved to {submission_file_path}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
