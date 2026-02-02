import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.utils import set_seed


def main():
    # 1. Setup and Configuration
    # We use a moderate number of epochs to balance speed and performance.
    # On an A100, 15 epochs on this dataset is very fast (approx 10-15 mins).
    N_EPOCHS = 15

    set_seed(Config.SEED)

    print(f"Initializing run with {N_EPOCHS} epochs...")

    # 2. Data Loading
    # Load full dataset (load_cached_data=True uses pre-computed npy files if available)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Training
    trainer = Trainer()
    # fit() automatically loads the best model weights upon completion
    trainer.fit(train_loader, val_loader, epochs=N_EPOCHS)

    # 4. Validation Assessment
    # Re-run validation to get the exact final metric on the best model
    val_loss, val_auc = trainer.validate(val_loader)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Get predictions and targets for the validation set
    trainer.model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(trainer.device)
            output = trainer.model(data)
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(output).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Load metadata to link back to file properties
    if os.path.exists(Config.VAL_CSV):
        df_val = pd.read_csv(Config.VAL_CSV)

        # Extract features for correlation: Duration
        # We read headers of the files to get duration
        durations = []

        # Note: df_val order matches val_loader because shuffle=False in val_loader
        # and we didn't drop samples.
        for _, row in df_val.iterrows():
            full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])
            try:
                # Fast header read
                info = sf.info(full_path)
                durations.append(info.duration)
            except Exception:
                durations.append(0.0)

        # Create a DataFrame for correlation analysis
        df_analysis = pd.DataFrame(
            {"error": errors, "duration": durations, "label": all_targets}
        )

        # Compute Correlations
        corr_matrix = df_analysis.corr()

        corr_duration = corr_matrix.loc["error", "duration"]
        corr_label = corr_matrix.loc["error", "label"]

        print(f"Correlation (Error vs Duration): {corr_duration:.8f}")
        print(f"Correlation (Error vs Label): {corr_label:.8f}")

        # Simple insight
        if abs(corr_duration) > 0.1:
            print("Insight: Error is somewhat correlated with audio duration.")
        if abs(corr_label) > 0.1:
            print(
                "Insight: Error is somewhat correlated with class label (class imbalance effect)."
            )

    else:
        print("Validation metadata not found, skipping detailed feature correlation.")

    # 6. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 0.9942618903292241

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} > {THRESHOLD}. Generating submission...")

        # Generate predictions
        clips, probs = trainer.predict(test_loader)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"clip": clips, "probability": probs})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Verify file creation
        if os.path.exists(Config.SUBMISSION_PATH):
            print("Submission file verified.")
    else:
        print(
            f"\nValidation metric {val_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
