import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.cuda import amp

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import get_model
from library.train import Trainer
from library.predict import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution.
    # 1 epoch on the full dataset is sufficient for a baseline and fits within the time limit.
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = None

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Training Pipeline
    # -------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training Loop...")
    # trainer.fit() handles training, validation monitoring, checkpointing,
    # and automatically generates the submission file at the end.
    trainer.fit(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("Starting Validation Assessment and Failure Analysis...")

    # Load validation metadata to get features for analysis (e.g., region_id)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Get DataLoaders
    # We use load_cached_data=True to utilize cached weights if available
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the best model saved during training for analysis
    model = get_model(pretrained=False)
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(
            Config.MODEL_CHECKPOINT_PATH, map_location=Config.DEVICE
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Checkpoint not found. Analysis will use current model state.")

    model.eval()
    device = Config.DEVICE

    all_preds = []
    all_labels = []

    # Optimized Inference Loop
    # - No gradients to save memory and computation
    # - AMP enabled for speed
    # - Non-blocking data transfer
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)

            with amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # 3.1 Calculate and Print Final Validation Metric
    final_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Final Validation Metric: {final_f1}")

    # 3.2 Failure Analysis
    # Define error: 1 if prediction is incorrect, 0 if correct
    errors = (y_pred != y_true).astype(int)

    # Add errors to the dataframe (order is preserved as shuffle=False in val_loader)
    if len(val_df) == len(errors):
        val_df["error"] = errors

        # Calculate correlation with region_id
        if "region_id" in val_df.columns:
            # Calculate correlation between the categorical region_id and binary error
            correlation = val_df["region_id"].corr(val_df["error"])
            print(f"Correlation between Error and Region ID: {correlation}")
        else:
            print("region_id column missing in validation metadata.")
    else:
        print(
            f"Mismatch in validation set size: DF {len(val_df)} vs Preds {len(errors)}"
        )

    # -------------------------------------------------------------------------
    # 4. Submission Verification
    # -------------------------------------------------------------------------
    # Trainer.fit() calls generate_submission, but we verify existence here.
    if not os.path.exists(Config.SUBMISSION_PATH):
        print("Submission file not found. Generating manually...")
        generate_submission()
    else:
        print(f"Submission successfully generated at {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
