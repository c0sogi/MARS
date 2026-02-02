import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss
from library.dataset import get_data_loaders
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We use the full dataset to ensure we can beat the target score,
    # but we optimize batch size and limit epochs to ensure it runs quickly on the A100.
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 1024  # A100 allows for larger batch size, speeding up training

    print(
        f"Starting execution with Config: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 2. Model Training
    # Initialize the Trainer
    trainer = Trainer()

    # Run the training loop
    # This handles data loading, training, validation checkpointing, and generates an initial submission
    trainer.fit(load_cached_data=True)

    # 3. Validation Assessment & Failure Analysis
    print("\n=== Starting Validation Assessment ===")

    # Retrieve data loaders to access the validation set
    # load_cached_data=True ensures we use the preprocessed data on disk
    _, val_loader, test_loader = get_data_loaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Load the best model checkpoint saved during training
    model_path = os.path.join(Config.OUTPUT_DIR, "model.pth")
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        trainer.model.load_state_dict(
            torch.load(model_path, map_location=trainer.device)
        )
    else:
        print("Warning: No model checkpoint found. Using current model weights.")

    trainer.model.eval()

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    # Metric accumulators
    total_masked_loss = 0.0
    total_mask_count = 0.0

    # Inference Loop (No Grad for speed)
    with torch.no_grad():
        for x, u_out, y in val_loader:
            x = x.to(trainer.device)
            u_out = u_out.to(trainer.device)
            y = y.to(trainer.device)

            # Forward pass
            preds = trainer.model(x)

            # Calculate Masked L1 Error for Metric
            # Mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
            mask = 1 - u_out
            abs_error = torch.abs(preds - y)
            masked_error = abs_error * mask

            total_masked_loss += masked_error.sum().item()
            total_mask_count += mask.sum().item()

            # Store data for failure analysis (move to CPU)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    # Compute Final Metric
    final_metric = total_masked_loss / total_mask_count if total_mask_count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Flatten collected data
    flat_preds = np.concatenate(all_preds).flatten()
    flat_targets = np.concatenate(all_targets).flatten()
    flat_u_out = np.concatenate(all_u_out).flatten()

    # Inputs: (N_batches, Batch, Seq, Feats) -> (Total_Steps, Feats)
    flat_inputs = np.concatenate(all_inputs)
    flat_inputs = flat_inputs.reshape(-1, flat_inputs.shape[-1])

    # Calculate absolute error
    errors = np.abs(flat_preds - flat_targets)

    # Filter for Inspiratory Phase only (u_out == 0)
    # The expiratory phase is not scored and physics differ, so we analyze failures where it matters.
    insp_mask = flat_u_out == 0
    insp_errors = errors[insp_mask]
    insp_inputs = flat_inputs[insp_mask]

    # Construct DataFrame for correlation analysis
    # Feature mapping: Continuous Features + R_cat + C_cat
    feature_names = Config.CONT_FEATURES + ["R_cat", "C_cat"]

    analysis_df = pd.DataFrame(insp_inputs, columns=feature_names)
    analysis_df["error_magnitude"] = insp_errors

    # Compute correlation
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features (Inspiratory Phase):")
    print(correlations)

    # 5. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.5275861736753809
    submission_file = "./submission/submission.csv"

    if final_metric < THRESHOLD:
        print(f"\nSuccess: Metric {final_metric} is below threshold {THRESHOLD}.")
        # Trainer.fit() automatically calls predict() which saves to ./submission/submission.csv
        # We verify it exists
        if os.path.exists(submission_file):
            print(f"Submission file confirmed at {submission_file}")
        else:
            print("Submission file missing. Regenerating...")
            trainer.predict(test_loader)
    else:
        print(f"\nFailure: Metric {final_metric} is above threshold {THRESHOLD}.")
        # Remove the submission file if it exists to comply with "If and only if"
        if os.path.exists(submission_file):
            os.remove(submission_file)
            print(f"Removed invalid submission file at {submission_file}")


if __name__ == "__main__":
    main()
