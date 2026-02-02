import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.utils import set_seed, get_device, Logger
from library.data import get_dataloaders
from library.model import DVSEModel, train_one_epoch, predict_and_submit

# ==========================================
# Constants
# ==========================================
METADATA_DIR = "./metadata"
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
WORK_DIR = "./working"
BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"
THRESHOLD_METRIC = 0.6978181818181817

# Hyperparameters for Fast Baseline
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4


def get_val_predictions(model, loader, device):
    """
    Runs inference on the validation set and aggregates View A/B predictions.
    Returns: (targets, probabilities) aligned with the validation metadata.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    # Convert to numpy
    probs_np = np.array(all_probs)
    targets_np = np.array(all_targets)

    # Reshape to (N_patients, 2) to group View A and View B
    # The validation loader is sequential: [P1_A, P1_B, P2_A, P2_B, ...]
    if len(probs_np) % 2 != 0:
        # Fallback if data length is odd (unexpected)
        return targets_np, probs_np

    probs_reshaped = probs_np.reshape(-1, 2)
    targets_reshaped = targets_np.reshape(-1, 2)

    # Average probabilities across views
    avg_probs = probs_reshaped.mean(axis=1)
    # Targets are identical for both views, take the first column
    patient_targets = targets_reshaped[:, 0]

    return patient_targets, avg_probs


def run_failure_analysis(val_targets, val_probs):
    """
    Calculates error and correlates it with metadata features.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # Load metadata to get features
    if not os.path.exists(VAL_META_PATH):
        print("Validation metadata not found. Skipping analysis.")
        return

    df_val = pd.read_parquet(VAL_META_PATH)

    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    # Extract features: Slice counts per modality
    features = {}
    for mod in ["flair", "t1w", "t1wce", "t2w"]:
        col = f"{mod}_paths"
        # Count number of files in the list
        counts = df_val[col].apply(lambda x: len(x) if x is not None else 0)
        features[f"{mod}_count"] = counts.values

    print("Correlation between Absolute Error and Metadata Features:")
    print("-" * 50)

    for name, values in features.items():
        if len(values) == len(errors):
            # Calculate correlation
            corr = np.corrcoef(errors, values)[0, 1]
            print(f"{name:<15}: {corr:.4f}")
        else:
            print(f"{name:<15}: Length mismatch ({len(values)} vs {len(errors)})")


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    logger = Logger()

    os.makedirs(WORK_DIR, exist_ok=True)

    # 2. Data Loading
    logger.section("Loading Data")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True
    )

    # 3. Model Initialization
    logger.section("Initializing DVSE Model")
    model = DVSEModel(
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0

    logger.section(f"Starting Training ({EPOCHS} epochs)")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate (Ensemble)
        val_targets, val_probs = get_val_predictions(model, val_loader, device)
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auc = 0.5

        logger.log(
            f"Epoch {epoch}/{EPOCHS} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # 5. Final Evaluation & Analysis
    logger.section("Final Evaluation")

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Get predictions from best model
    val_targets, val_probs = get_val_predictions(model, val_loader, device)
    final_auc = roc_auc_score(val_targets, val_probs)

    # REQUIRED: Print exact metric format
    print(f"Final Validation Metric: {final_auc}")

    # Run Failure Analysis
    run_failure_analysis(val_targets, val_probs)

    # 6. Submission
    if final_auc > THRESHOLD_METRIC:
        logger.section("Generating Submission")
        predict_and_submit(model, test_loader, output_path=SUBMISSION_PATH)
    else:
        logger.log(
            f"Final AUC {final_auc} does not exceed threshold {THRESHOLD_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
