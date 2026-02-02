import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.train import Trainer
from library.utils import seed_everything, calculate_weighted_log_loss
from library.data import get_dataloaders
from library.model import CervicalFractureNet


def analyze_failures(val_df, preds, targets):
    """
    Analyzes prediction errors on the validation set.
    Calculates the correlation between error magnitude and target presence.
    """
    # Calculate binary cross entropy per sample (averaged across the 8 classes)
    # preds: (N, 8), targets: (N, 8)
    epsilon = 1e-15
    preds = np.clip(preds, epsilon, 1 - epsilon)

    # Compute error (absolute difference or log loss per sample)
    # Using Log Loss per sample as the error magnitude
    # Loss = - (y * log(p) + (1-y) * log(1-p))
    sample_losses = -(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
    mean_sample_loss = np.mean(sample_losses, axis=1)  # Average loss per study

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["error_magnitude"] = mean_sample_loss

    # Calculate correlation between error and having a fracture (patient_overall)
    if "patient_overall" in analysis_df.columns:
        corr = analysis_df["error_magnitude"].corr(analysis_df["patient_overall"])
        print(f"Correlation between Error Magnitude and 'patient_overall': {corr:.4f}")

        # Check correlation with specific fracture types if they exist
        for col in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            if col in analysis_df.columns:
                c_corr = analysis_df["error_magnitude"].corr(analysis_df[col])
                # Only print if significant or for debugging
                # print(f"Correlation Error vs {col}: {c_corr:.4f}")


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    study_ids = []
    all_preds = []

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            # batch['study_id'] is a tuple/list
            ids = batch["study_id"]

            outputs = model(images)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs["fracture_logits"]).cpu().numpy()

            study_ids.extend(ids)
            all_preds.append(probs)

    all_preds = np.concatenate(all_preds, axis=0)

    # Column mapping matches the model output order defined in Dataset/Model
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    row_ids = []
    fractured_vals = []

    for i, uid in enumerate(study_ids):
        preds = all_preds[i]
        for j, col in enumerate(cols):
            # Format: StudyInstanceUID_Label
            if col == "patient_overall":
                rid = f"{uid}_patient_overall"
            else:
                rid = f"{uid}_{col}"

            row_ids.append(rid)
            fractured_vals.append(preds[j])

    submission_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_vals})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Configuration
    config = Config()

    # Fast Baseline Settings
    config.EPOCHS = 5  # Small dataset, 5 epochs is fast enough
    config.BATCH_SIZE = 1  # Keep small for safety
    config.EFFECTIVE_BATCH_SIZE = 16
    config.GRAD_ACCUM_STEPS = config.EFFECTIVE_BATCH_SIZE // config.BATCH_SIZE

    # Ensure reproducibility
    seed_everything(config.SEED)

    # 2. Training
    print("Initializing Trainer...")
    trainer = Trainer(config)
    trainer.fit()

    # 3. Validation & Metric Calculation
    print("Running Validation...")
    # Load best model
    model = CervicalFractureNet(config)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)
    model.eval()

    _, val_loader, test_loader = get_dataloaders(config)

    val_preds = []
    val_targets = []
    val_study_ids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(config.DEVICE)
            fracture_labels = batch["fracture_labels"].to(config.DEVICE)
            ids = batch["study_id"]

            outputs = model(images)
            probs = torch.sigmoid(outputs["fracture_logits"]).cpu().numpy()
            targets = fracture_labels.cpu().numpy()

            val_preds.append(probs)
            val_targets.append(targets)
            val_study_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    # Reconstruct DataFrame for metric calculation function
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    y_true_data = {"StudyInstanceUID": val_study_ids}
    for i, col in enumerate(cols):
        y_true_data[col] = val_targets[:, i]
    y_true_df = pd.DataFrame(y_true_data)

    # Construct y_pred_df
    row_ids = []
    fractured_probs = []
    for i, uid in enumerate(val_study_ids):
        for j, col in enumerate(cols):
            if col == "patient_overall":
                rid = f"{uid}_patient_overall"
            else:
                rid = f"{uid}_{col}"
            row_ids.append(rid)
            fractured_probs.append(val_preds[i, j])

    y_pred_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_probs})

    final_metric = calculate_weighted_log_loss(y_true_df, y_pred_df)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    analyze_failures(y_true_df, val_preds, val_targets)

    # 5. Conditional Submission
    THRESHOLD = 0.15364714496434773

    # Note: For the purpose of this task, we will generate the submission
    # if the metric is good enough OR if it's close enough to be a valid attempt
    # in a real scenario (often thresholds are baselines).
    # However, strictly following the prompt:
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        submission_path = "./submission/submission.csv"
        generate_submission(model, test_loader, config.DEVICE, submission_path)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
