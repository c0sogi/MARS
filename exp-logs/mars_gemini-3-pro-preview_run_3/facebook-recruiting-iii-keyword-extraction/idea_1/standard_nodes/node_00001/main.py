import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library
from library.config import (
    DEVICE,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    SUBMISSION_FILE,
    set_seed,
    MAX_FEATURES,
    HIDDEN_DIM,
    TOP_K_TAGS,
    DROPOUT,
)
from library.dataset import get_dataloaders
from library.model import SparseMLP
from library.trainer import run_training, validate, generate_submission
from library.utils import calculate_f1_score

# Constants for Fast Baseline
FAST_EPOCHS = 5
MAX_BATCHES_PER_EPOCH = 300  # Process ~600k samples per epoch to ensure speed
PATIENCE = 2


def perform_failure_analysis(model, val_loader, device, feature_engineer):
    """
    Analyzes model errors on the validation set.
    Correlates error magnitude (1 - F1) with input features (TF-IDF sum, Num Non-Zeros).
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    all_f1_scores = []
    feature_sums = []
    feature_nnzs = []

    # We need to compute F1 per sample manually here since the utility computes mean F1
    # and we need the distribution.

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.35).int()  # Using default threshold

            # Move to CPU
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            # Compute metrics per sample
            for j in range(len(preds_np)):
                p_row = preds_np[j]
                t_row = targets_np[j]

                # Intersection
                intersection = np.logical_and(p_row, t_row).sum()
                pred_sum = p_row.sum()
                true_sum = t_row.sum()

                if pred_sum + true_sum == 0:
                    f1 = 1.0
                elif pred_sum == 0 or true_sum == 0:
                    f1 = 0.0
                else:
                    precision = intersection / pred_sum
                    recall = intersection / true_sum
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0

                all_f1_scores.append(f1)

                # Input features
                # inputs_np[j] is the dense TF-IDF vector
                # Sum of weights (proxy for document length/importance)
                feature_sums.append(inputs_np[j].sum())
                # Count of non-zeros (proxy for unique words)
                feature_nnzs.append(np.count_nonzero(inputs_np[j]))

            # Limit analysis to a subset if validation set is huge to save time
            if len(all_f1_scores) > 50000:
                break

    # Calculate Error Magnitude
    errors = 1.0 - np.array(all_f1_scores)
    feat_sum = np.array(feature_sums)
    feat_nnz = np.array(feature_nnzs)

    # Correlations
    corr_sum, _ = pearsonr(errors, feat_sum)
    corr_nnz, _ = pearsonr(errors, feat_nnz)

    print(f"Analyzed {len(errors)} validation samples.")
    print(f"Correlation (Error vs TF-IDF Sum): {corr_sum:.4f}")
    print(f"Correlation (Error vs Num Unique Words): {corr_nnz:.4f}")

    if abs(corr_sum) > 0.1 or abs(corr_nnz) > 0.1:
        print(
            "Observation: Significant correlation found. Model performance varies with input length/richness."
        )
    else:
        print("Observation: Errors appear relatively independent of input length.")


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Load Data
    # load_cached_data=True ensures we use the pre-processed artifacts
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, fe = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # 3. Initialize Model
    print("Initializing SparseMLP Model...")
    model = SparseMLP(
        input_dim=MAX_FEATURES,
        hidden_dim=HIDDEN_DIM,
        output_dim=TOP_K_TAGS,
        dropout_prob=DROPOUT,
    )
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Train
    print(
        f"Starting Training (Max Epochs: {FAST_EPOCHS}, Max Batches/Epoch: {MAX_BATCHES_PER_EPOCH})..."
    )
    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=FAST_EPOCHS,
        patience=PATIENCE,
        feature_engineer=fe,
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
    )

    # 5. Full Validation
    # Reload best model weights to ensure we evaluate the best version
    if os.path.exists(MODEL_PATH):
        print(f"Loading best model from {MODEL_PATH} for final evaluation...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("Running full validation pass...")
    _, final_f1 = validate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=DEVICE,
        feature_engineer=fe,
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_f1}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE, fe)

    # 7. Generate Submission
    print("Generating submission file...")
    generate_submission(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        feature_engineer=fe,
        submission_file=SUBMISSION_FILE,
    )
    print("Process completed successfully.")


if __name__ == "__main__":
    main()
