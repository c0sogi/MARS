import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train import Trainer
from library import utils


def main():
    # 1. Setup and Configuration Overrides
    utils.set_seed()

    # Adjust epochs to ensure execution finishes within 2 hours
    # A100 is fast, but we want to be safe.
    # ~144k images. 1 epoch ~5-10 mins.
    # 1 epoch stage 1, 3 epochs stage 2 = ~30-40 mins total.
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE2 = 3

    # Ensure we are not in debug mode for the final run to get good performance
    Config.DEBUG = False

    print("Initializing Trainer...")
    trainer = Trainer(debug=Config.DEBUG)

    # 2. Training Execution
    # Stage 1: Train Head
    trainer.run_stage1()

    # Stage 2: Fine-tune Backbone
    trainer.run_stage2()

    # 3. Validation & Metric Calculation
    print("\n=== Performing Final Validation ===")

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        trainer.model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    else:
        print("Warning: No saved model found. Using current model state.")

    trainer.model.eval()
    val_loader = trainer.val_loader

    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)

            logits = trainer.model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate Metric
    final_f1 = utils.calculate_macro_f1(all_labels, all_preds)
    print(f"Final Validation Metric: {final_f1}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load metadata to correlate errors with features
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure lengths match (loader should be sequential and match metadata)
    if len(val_metadata) != len(all_preds):
        print(
            f"Warning: Metadata length ({len(val_metadata)}) does not match predictions ({len(all_preds)}). Skipping detailed correlation."
        )
    else:
        # Create binary error vector (1 for incorrect, 0 for correct)
        errors = (np.array(all_preds) != np.array(all_labels)).astype(int)
        val_metadata["error_magnitude"] = errors

        # Select numerical columns for correlation
        numeric_cols = val_metadata.select_dtypes(include=[np.number]).columns

        # Remove target columns or IDs from correlation check if present
        cols_to_check = [
            c
            for c in numeric_cols
            if c not in ["Category", "error_magnitude", "Unnamed: 0"]
        ]

        print(
            "Correlation between Error Magnitude (Misclassification) and Input Features:"
        )
        if cols_to_check:
            correlations = val_metadata[cols_to_check].corrwith(
                val_metadata["error_magnitude"]
            )
            print(correlations)
        else:
            print("No numerical features found for correlation analysis.")

    # 5. Conditional Submission
    THRESHOLD = 0.3840415638913998

    if final_f1 > THRESHOLD:
        print(
            f"\nValidation score ({final_f1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation score ({final_f1}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
