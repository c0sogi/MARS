import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
import scipy.stats
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


# ------------------------------------------------------------------------------
# 1. Environment Patching (Suppress Progress Bars)
# ------------------------------------------------------------------------------
# Create a dummy tqdm class that does nothing but iterate
class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    @classmethod
    def write(cls, s):
        print(s)


# Inject the dummy class into sys.modules to intercept imports in library files
# We create a fake module object
tqdm_module = type("Module", (), {"tqdm": DummyTqdm})
sys.modules["tqdm"] = tqdm_module

# ------------------------------------------------------------------------------
# 2. Imports from Library
# ------------------------------------------------------------------------------
from library.config import Config
from library.utils import seed_everything
from library.data import BrainTumorDataset
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.predict import predict_submission


def main():
    # Filter warnings
    warnings.filterwarnings("ignore")

    # Set reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 3. Configuration Overrides
    # --------------------------------------------------------------------------
    # Limit epochs for fast baseline execution
    Config.EPOCHS = 15
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Starting training pipeline...")
    run_training()

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nStarting validation assessment...")

    # Load Validation Data
    df_val = pd.read_csv(Config.VAL_CSV)
    val_dataset = BrainTumorDataset(df_val, phase="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = Config.DEVICE
    model = AsymmetricEfficientNet().to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model checkpoint not found. Using random weights.")

    model.eval()

    # Inference Loop
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Sigmoid to get probabilities
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())

    # Compute Final Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(np.array(all_preds) - np.array(all_targets))

    # Extract Metadata Features for Correlation
    # We correlate error with slice counts and anchor position
    feature_rows = []
    for i, row in df_val.iterrows():
        subject_id = row["BraTS21ID"]

        # Get Anchor ID used by the dataset
        anchor_id = val_dataset.roi_map.get(subject_id, -1)

        features = {"anchor_id": anchor_id}

        # Get file counts for each modality
        for mod in Config.MODALITIES:
            path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            if os.path.exists(path):
                features[f"{mod}_count"] = len(os.listdir(path))
            else:
                features[f"{mod}_count"] = 0

        feature_rows.append(features)

    df_features = pd.DataFrame(feature_rows)

    print("Correlation between Error Magnitude and Input Features:")
    for col in df_features.columns:
        if df_features[col].std() > 0:  # Skip constant columns
            corr, _ = scipy.stats.pearsonr(df_features[col], errors)
            print(f"  {col}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6321818181818182

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_submission()
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
