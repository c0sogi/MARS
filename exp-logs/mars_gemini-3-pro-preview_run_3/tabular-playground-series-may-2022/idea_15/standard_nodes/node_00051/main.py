import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import necessary components from the provided library
from library.config import Config
from library.train_utils import train_model, generate_submission
from library.model import ManufacturingMLP
from library.data_utils import get_dataloaders


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(model, val_loader, device, cont_cols):
    """
    Analyzes the model's failure modes on the validation set.
    Calculates the correlation between prediction error magnitude and continuous input features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_errors = []
    all_cont_features = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            # Calculate absolute error
            error = torch.abs(y - probs)

            all_errors.append(error.cpu().numpy())
            all_cont_features.append(x_cont.cpu().numpy())

    all_errors = np.vstack(all_errors).flatten()
    all_cont_features = np.vstack(all_cont_features)

    # Create a DataFrame to compute correlations
    # We focus on continuous features for meaningful correlation analysis
    df_analysis = pd.DataFrame(all_cont_features, columns=cont_cols)
    df_analysis["error_magnitude"] = all_errors

    # Compute correlation with error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation to find most impactful features
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error Magnitude:")
    print(sorted_corr.head(5))

    return sorted_corr


def main():
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train the Model
    # We use the full configuration (30 epochs, full dataset) to ensure we meet the high AUC requirement.
    # The A100 GPU allows this to complete well within the time limit.
    print("Starting Training Pipeline...")
    train_model(load_cached_data=True)

    # 3. Load Best Model for Final Evaluation
    # We reload the model from disk to ensure we are using the best checkpoint saved during training.
    print("\nLoading best model for evaluation...")

    # Re-initialize dataloaders to get metadata and validation set
    _, val_loader, _, num_continuous, vocab_sizes = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Initialize model architecture
    model = ManufacturingMLP(
        num_continuous=num_continuous,
        categorical_vocab_sizes=vocab_sizes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        output_dim=Config.OUTPUT_DIM,
    ).to(device)

    # Load weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model file {Config.MODEL_SAVE_PATH} not found.")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 4. Compute Final Validation Metric
    print("Computing Final Validation Metric on full hold-out set...")
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            all_targets.append(y.numpy())  # y is from dataset (CPU)
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds).flatten()

    final_auc = roc_auc_score(all_targets, all_preds)

    # Print metric with full precision as required
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    # Load metadata to get feature names
    metadata = np.load(Config.METADATA_CACHE, allow_pickle=True).item()
    cont_cols = metadata["cont_cols"]

    perform_failure_analysis(model, val_loader, device, cont_cols)

    # 6. Conditional Submission
    # Threshold defined in the task
    THRESHOLD = 0.9971550270448856

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation metric {final_auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
