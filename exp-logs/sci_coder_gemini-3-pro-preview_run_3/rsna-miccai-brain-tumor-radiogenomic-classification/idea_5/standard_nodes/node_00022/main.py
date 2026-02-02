import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config, seed_everything
from library.train import run_training
from library.predict import generate_submission
from library.dataset import get_dataloader
from library.model import VFPNet


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    # The dataset is small (418 training samples), so 10 epochs is sufficient and fast.
    Config.NUM_EPOCHS = 10

    # 2. Run Training
    print("Starting training pipeline...")
    run_training()

    # 3. Validation Assessment
    print("\nStarting Validation Assessment...")
    device = torch.device(Config.DEVICE)

    # Initialize model structure (pretrained=False as we load custom weights)
    model = VFPNet(num_classes=1, pretrained=False)

    # Load best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model weights not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get validation dataloader
    val_loader = get_dataloader("val", shuffle=False, load_cached_data=True)

    all_targets = []
    all_preds = []

    # Inference loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Calculate Metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load validation metadata to retrieve input features (slice counts)
    val_df = pd.read_parquet(Config.VAL_META_PATH)

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Construct DataFrame for correlation analysis
    analysis_df = pd.DataFrame()

    # Extract slice counts for each modality
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate count of slices (handle None/empty lists if any)
        analysis_df[f"{mod}_count"] = val_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )

    # Add error to dataframe
    analysis_df["error"] = errors

    # Calculate correlations
    print("Correlation between Error Magnitude and Input Features:")
    correlations = analysis_df.corr()["error"].drop("error")
    print(correlations)

    # 5. Submission Generation
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        generate_submission(weights_path=best_model_path)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
