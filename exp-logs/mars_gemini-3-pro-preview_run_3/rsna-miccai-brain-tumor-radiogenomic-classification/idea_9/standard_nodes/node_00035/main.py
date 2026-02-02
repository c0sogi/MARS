import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import load_dataset
from library.model import Stabilized25DNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Execution Device: {device}")

    # 2. Training
    # We use the config defaults (15 epochs) which is efficient for this dataset size.
    # This ensures a quick baseline execution.
    print("Starting Training...")
    run_training(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

    # 3. Validation & Metric Calculation
    print("Starting Validation...")

    # Load Validation Dataset
    val_dataset = load_dataset("val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = Stabilized25DNet().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Targets are loaded as float32 tensors

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Final Metric
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    # Load metadata to retrieve input features (slice counts)
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    val_df = pd.read_parquet(val_meta_path)

    # Construct analysis dataframe
    # Note: The dataset loader preserves the order of the dataframe
    analysis_df = pd.DataFrame(
        {"error": errors, "target": all_targets, "pred": all_probs}
    )

    # Extract meta-features: slice counts per modality
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Count number of files in the list
        counts = val_df[col_name].apply(lambda x: len(x) if x is not None else 0)
        analysis_df[f"{mod}_count"] = counts.values

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    threshold = 0.6978181818181817

    if val_auc > threshold:
        print(
            f"Validation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE)
    else:
        print(
            f"Validation AUC ({val_auc}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
