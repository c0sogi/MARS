import os
import torch
import pandas as pd
import numpy as np
from library.config import (
    SEED,
    DEVICE,
    NUM_EPOCHS,
    WORKING_DIR,
    IDX_TO_LABEL,
    VAL_METADATA_PATH,
)
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import MultiScaleEfficientNet
from library.trainer import run_training


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    # Use cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    # Run training using the provided trainer utility
    # We use the config's NUM_EPOCHS (30) which is optimized for this task/hardware
    print("Starting training...")
    _ = run_training(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # 4. Load Best Model for Evaluation
    # run_training returns the model state at the last epoch.
    # We must load the best checkpoint saved during training for accurate validation.
    model = MultiScaleEfficientNet()
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Error: Best model checkpoint not found!")
        return

    model.to(DEVICE)
    model.eval()

    # 5. Validation Assessment
    print("Running validation inference...")
    val_preds = []
    val_targets = []
    val_fnames = []

    with torch.no_grad():
        for inputs, targets, fnames in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            # Get class predictions
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
            val_fnames.extend(fnames)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Compute and print the required metric
    val_acc = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Construct a DataFrame to analyze errors
    df_analysis = pd.DataFrame(
        {
            "fname": val_fnames,
            "target": val_targets,
            "pred": val_preds,
            "error": (val_preds != val_targets).astype(int),
        }
    )

    # Calculate correlation between Error and Target Label (as a proxy for class difficulty)
    # This reveals if specific classes are systematically contributing to the error rate.
    corr_label = df_analysis["error"].corr(df_analysis["target"])
    print(f"Correlation between Error and Target Label Index: {corr_label}")

    # 7. Submission
    threshold = 0.9853666694539677
    if val_acc > threshold:
        print(
            f"Validation metric {val_acc} exceeds threshold {threshold}. Generating submission..."
        )

        test_preds = []
        test_fnames_list = []

        with torch.no_grad():
            for inputs, _, fnames in test_loader:
                inputs = inputs.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                test_preds.extend(preds.cpu().numpy())
                test_fnames_list.extend(fnames)

        # Map integer indices back to string labels
        test_labels_str = [IDX_TO_LABEL[idx] for idx in test_preds]

        # Create submission DataFrame
        sub_df = pd.DataFrame({"fname": test_fnames_list, "label": test_labels_str})

        # Save to file
        os.makedirs("submission", exist_ok=True)
        submission_path = "submission/submission.csv"
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {val_acc} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
