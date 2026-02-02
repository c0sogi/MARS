import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss_score
from library.data import get_dataloaders
from library.model import CervicalSpineMILModel
from library.engine import fit, inference


def main():
    # --- 1. Setup ---
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- 2. Data Loading ---
    # Loaders handle caching and preprocessing automatically.
    # We use the default batch size (8) and num_workers (4) from Config.
    train_loader, val_loader, test_loader = get_dataloaders()

    # --- 3. Model Initialization ---
    model = CervicalSpineMILModel(pretrained=True)
    model = model.to(device)

    # --- 4. Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # --- 5. Training ---
    # Run for 10 epochs to ensure convergence without box-guidance.
    # The fit function handles the training loop, validation monitoring, and saving the best model.
    print("Starting training...")
    fit(model, train_loader, val_loader, optimizer, device, epochs=10)

    # --- 6. Final Evaluation ---
    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_targets = []
    study_ids = []

    # Perform inference on validation set to compute the final metric
    with torch.no_grad():
        for images, targets, s_ids in val_loader:
            images = images.to(device)

            # Forward pass
            instance_logits = model(images)

            # Global Max Pooling Aggregation
            # Shape: (B, S, 7) -> (B, 7)
            pooled_logits, _ = torch.max(instance_logits, dim=1)

            # Derive patient_overall logit: max(C1..C7)
            # Shape: (B, 1)
            patient_logit, _ = torch.max(pooled_logits, dim=1, keepdim=True)

            # Concatenate to get 8 columns: [C1...C7, patient_overall]
            global_logits = torch.cat([pooled_logits, patient_logit], dim=1)

            # Convert to probabilities
            probs = torch.sigmoid(global_logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            study_ids.extend(s_ids)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Calculate and print the required metric
    final_metric = get_weighted_log_loss_score(y_pred, y_true)
    print(f"Final Validation Metric: {final_metric}")

    # --- 7. Failure Analysis ---
    print("Performing failure analysis...")

    # Calculate per-sample weighted log loss (error magnitude)
    # Weights: 1.0 for C1-C7, 7.0 for patient_overall
    weights = np.array([1.0] * 7 + [7.0])
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    # Loss matrix: (N, 8)
    loss_matrix = -weights * (
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )
    # Mean loss per sample
    sample_losses = np.mean(loss_matrix, axis=1)

    # Load metadata to retrieve input features for correlation
    val_df = pd.read_csv(Config.VAL_METADATA)
    # Ensure metadata is aligned with the prediction order
    val_df = val_df.set_index("StudyInstanceUID").reindex(study_ids).reset_index()

    # Feature 1: Fracture Presence (Target Class)
    feat_fracture = val_df["patient_overall"].values

    # Feature 2: Scan Depth (Number of slices in the directory)
    feat_depth = []
    for rel_path in val_df["image_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            # Fast count of files in the directory
            count = len(
                [
                    name
                    for name in os.listdir(full_path)
                    if os.path.isfile(os.path.join(full_path, name))
                ]
            )
        except Exception:
            count = 0
        feat_depth.append(count)
    feat_depth = np.array(feat_depth)

    # Calculate Correlations using Numpy (Pearson)
    if len(sample_losses) > 1:
        # Correlation with Fracture Presence
        if np.std(feat_fracture) > 1e-9:
            corr_fracture = np.corrcoef(sample_losses, feat_fracture)[0, 1]
            print(f"Correlation (Error vs Fracture Presence): {corr_fracture}")
        else:
            print("Correlation (Error vs Fracture Presence): N/A (Constant feature)")

        # Correlation with Scan Depth
        if np.std(feat_depth) > 1e-9:
            corr_depth = np.corrcoef(sample_losses, feat_depth)[0, 1]
            print(f"Correlation (Error vs Scan Depth): {corr_depth}")
        else:
            print("Correlation (Error vs Scan Depth): N/A (Constant feature)")

    # --- 8. Submission ---
    THRESHOLD = 0.12231192492082398

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure the submission directory exists
        os.makedirs("./submission", exist_ok=True)

        # Override Config path to match the specific output requirement
        Config.SUBMISSION_PATH = "./submission/submission.csv"

        # Run inference on test set
        inference(model, test_loader, device)
    else:
        print(
            f"Metric ({final_metric}) is not lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
