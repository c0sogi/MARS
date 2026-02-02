import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library functions
from library.utils import seed_everything
from library.model import GNHRNet
from library.data_loader import get_dataloaders
from library.train_eval import run_training, predict_and_submit

# Configuration
SEED = 42
EPOCHS = 15  # Fast baseline: sufficient epochs for the small dataset size
BATCH_SIZE = 8
LR = 1e-4
THRESHOLD = 0.6978181818181817
CHECKPOINT_PATH = "./working/idea_34/best_model.pth"
SUBMISSION_DIR = "./submission"


def perform_failure_analysis(val_loader, model, device):
    """
    Evaluates the model on the validation set, calculates the final metric,
    and correlates prediction errors with input meta-features (slice counts).
    """
    print("\nPerforming Failure Analysis...")

    model.eval()
    all_targets = []
    all_preds = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, targets, ids in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_targets.extend(targets.numpy().flatten())
            all_preds.extend(probs)
            all_ids.extend(ids)

    # Calculate Final Metric
    try:
        final_metric = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    # Load metadata to retrieve slice counts
    df_val = pd.read_parquet("./metadata/val.parquet")

    # Construct DataFrame for analysis
    results_df = pd.DataFrame(
        {
            "BraTS21ID": [str(x) for x in all_ids],
            "target": all_targets,
            "pred": all_preds,
        }
    )
    results_df["error"] = np.abs(results_df["target"] - results_df["pred"])

    # Ensure ID types match for merging
    df_val["BraTS21ID"] = df_val["BraTS21ID"].astype(str)

    # Merge prediction results with metadata
    analysis_df = pd.merge(results_df, df_val, on="BraTS21ID", how="left")

    modalities = ["flair", "t1w", "t1wce", "t2w"]
    print("Correlation between Error Magnitude and Slice Counts:")

    for mod in modalities:
        col_path = f"{mod}_paths"
        if col_path in analysis_df.columns:
            # Calculate slice count for the modality
            analysis_df[f"{mod}_count"] = analysis_df[col_path].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Calculate correlation
            if analysis_df[f"{mod}_count"].std() > 0:
                corr = analysis_df[f"{mod}_count"].corr(analysis_df["error"])
                print(f"Feature: {mod}_count | Correlation: {corr:.6f}")
            else:
                print(f"Feature: {mod}_count | Correlation: NaN (Constant feature)")

    return final_metric


def main():
    # 1. Setup Environment
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # 2. Training Phase
    print("Starting Training Pipeline...")
    # run_training handles the loop, validation, and saving the best model
    _ = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=5,
        num_workers=4,
        load_cached_data=True,
        limit_data=None,  # Use full training set (~400 samples)
        seed=SEED,
    )

    # 3. Validation & Failure Analysis Phase
    # Reload validation data loader
    dataloaders = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)
    val_loader = dataloaders["val"]

    # Initialize model architecture
    model = GNHRNet(
        model_name="efficientnet_b0", pretrained=False, in_chans=64, num_classes=1
    )

    # Load the best checkpoint saved during training
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded best model from {CHECKPOINT_PATH}")
    else:
        print("Warning: Checkpoint not found. Using random initialization.")

    model = model.to(device)

    # Execute analysis
    final_metric = perform_failure_analysis(val_loader, model, device)

    # 4. Submission Phase
    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")
        predict_and_submit(
            batch_size=BATCH_SIZE,
            num_workers=4,
            load_cached_data=True,
            output_dir=SUBMISSION_DIR,
        )
    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
