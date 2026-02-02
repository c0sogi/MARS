import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed, get_device, log_message, load_checkpoint
from library.data import get_dataloaders
from library.model import ModalityGroupedEfficientNet
from library.train import train_model


def run():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.EPOCHS = 10  # Increased to allow scheduler to work (Cite Lesson 00007)

    set_seed(Config.SEED)
    device = get_device()

    # --------------------------------------------------------------------------
    # 2. Train Model
    # --------------------------------------------------------------------------
    # This will train, validate, and save the best model to Config.MODEL_PATH
    train_model(debug=False)

    # --------------------------------------------------------------------------
    # 3. Validation Inference
    # --------------------------------------------------------------------------
    log_message("Loading best model for validation and analysis...")

    # Re-initialize model architecture
    model = ModalityGroupedEfficientNet()
    model.to(device)

    # Load the best weights saved during training
    load_checkpoint(model, Config.MODEL_PATH)
    model.eval()

    # Get dataloaders (re-using the same function ensures consistency)
    _, val_loader, test_loader = get_dataloaders(debug=False)

    val_probs = []
    val_targets = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            val_probs.extend(probs)
            val_targets.extend(targets.numpy())

    val_probs = np.array(val_probs).flatten()
    val_targets = np.array(val_targets).flatten()

    # Calculate Metric
    try:
        val_auc = roc_auc_score(val_targets, val_probs)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    log_message("Performing Failure Analysis...")

    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Access metadata from the dataset
    val_dataset = val_loader.dataset
    val_df = val_dataset.df
    roi_map = val_dataset.roi_map

    # Extract features for correlation analysis
    analysis_data = []
    for idx, row in val_df.iterrows():
        sid = row["BraTS21ID"]

        # Feature 1: ROI Index (Anatomical position)
        roi_idx = roi_map.get(sid, 0)

        # Feature 2: Slice Count (Proxy for volume/resolution)
        # We check FLAIR folder
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Fast count of files
            num_slices = len(
                [name for name in os.listdir(flair_path) if name.endswith(".dcm")]
            )
        except Exception:
            num_slices = 0

        analysis_data.append(
            {"roi_idx": roi_idx, "num_slices": num_slices, "target": row["MGMT_value"]}
        )

    analysis_df = pd.DataFrame(analysis_data)
    analysis_df["error"] = errors

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Metadata Features:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6254545454545455

    if val_auc > THRESHOLD:
        log_message(
            f"Validation metric {val_auc} > {THRESHOLD}. Generating submission..."
        )

        test_probs = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.extend(probs)

        test_probs = np.array(test_probs).flatten()

        # Create submission DataFrame
        test_df = test_loader.dataset.df
        submission = pd.DataFrame(
            {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": test_probs}
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        log_message(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        log_message(
            f"Validation metric {val_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
