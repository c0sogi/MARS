import sys
import torch
import numpy as np
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import get_dataloaders
from library.train import train_model


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Enforce Fast Baseline constraints
    # Reducing epochs to 15 ensures the run completes quickly (well within 2 hours)
    # while allowing the aggressive scheduler to converge.
    Config.EPOCHS = 15

    print(f"Initializing Fast Baseline Run (Epochs: {Config.EPOCHS})...")

    # 2. Train Model
    # This handles the entire training loop, checkpointing, and restores the best model.
    model = train_model(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("Loading data for validation and analysis...")
    # Retrieve loaders to access validation data for metric calculation and failure analysis
    _, val_loader, test_loader, _ = get_dataloaders(load_cached_data=True)

    model.eval()
    model.to(device)

    val_preds_list = []
    val_targets_list = []
    val_features_list = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(target.cpu())
            # Keep features on CPU to avoid OOM during concatenation of large arrays
            val_features_list.append(data.cpu())

    val_preds = torch.cat(val_preds_list).numpy()
    val_targets = torch.cat(val_targets_list).numpy()
    val_features = torch.cat(val_features_list).numpy()

    # Calculate Final Metric
    accuracy = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation between Error and Features
    print("Performing Failure Analysis...")
    errors = (val_preds != val_targets).astype(int)

    n_features = val_features.shape[1]
    correlations = []

    for i in range(n_features):
        feat_col = val_features[:, i]
        # Handle constant features to avoid division by zero
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(
        "Top 10 features correlated with prediction error (Feature Index: Correlation):"
    )
    for idx, corr in correlations[:10]:
        print(f"Feature {idx}: {corr:.4f}")

    # 4. Submission Generation
    THRESHOLD = 0.9626291666666666

    if accuracy > THRESHOLD:
        print(f"Validation accuracy {accuracy} > {THRESHOLD}. Generating submission...")

        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for data, _, ids in test_loader:
                data = data.to(device)

                outputs = model(data)
                _, preds = torch.max(outputs, 1)

                test_preds_list.append(preds.cpu())
                test_ids_list.append(ids.cpu())

        final_preds = torch.cat(test_preds_list).numpy()
        final_ids = torch.cat(test_ids_list).numpy()

        # Map 0-based predictions (0-6) back to 1-based Cover_Type (1-7)
        final_preds = final_preds + 1

        save_submission(final_preds, final_ids)
    else:
        print(f"Validation accuracy {accuracy} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
