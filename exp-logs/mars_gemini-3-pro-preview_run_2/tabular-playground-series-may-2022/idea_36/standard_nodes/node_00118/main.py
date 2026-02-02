import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import load_and_preprocess_data
from library.model import ModalityScaledHybridSwiGLU
from library.train_eval import get_optimizer, train_one_epoch, validate, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Detect device
    device = get_device()

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load data with caching enabled for speed
    # This utilizes the pre-defined split in metadata
    train_loader, val_loader, test_loader = load_and_preprocess_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = ModalityScaledHybridSwiGLU().to(device)

    # --------------------------------------------------------------------------
    # 4. Training Loop (Fast Baseline)
    # --------------------------------------------------------------------------
    optimizer = get_optimizer(model)
    criterion = nn.BCEWithLogitsLoss()

    # Limit epochs to ensures execution within time limits while allowing convergence
    FAST_EPOCHS = 5
    best_auc = 0.0

    # Ensure model directory exists
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    for epoch in range(FAST_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # --------------------------------------------------------------------------
    # Load the best performing model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Compute metric on the full hold-out validation set
    _, final_val_auc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {final_val_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_errors = []
    all_cont_features = []

    # Collect predictions and features from validation set
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            logits = model(continuous, sequence)
            probs = torch.sigmoid(logits)

            # Calculate absolute error magnitude
            error = torch.abs(targets - probs)

            all_errors.append(error.cpu().numpy())
            all_cont_features.append(continuous.cpu().numpy())

    # Flatten and concatenate
    all_errors = np.concatenate(all_errors).flatten()
    all_cont_features = np.concatenate(all_cont_features, axis=0)

    # Reconstruct feature names for continuous columns (f_00..f_30 excluding f_27)
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(all_cont_features, columns=cont_cols)
    df_analysis["error_magnitude"] = all_errors

    # Calculate correlation between features and error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    print("Top 5 Features Correlated with Error Magnitude:")
    print(correlations.abs().sort_values(ascending=False).head(5))

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972883264620234

    if final_val_auc > THRESHOLD:
        print(f"\nValidation metric {final_val_auc} exceeds threshold {THRESHOLD}.")
        print("Generating submission...")

        # Generate predictions on test set
        preds = predict(model, test_loader, device)

        # Load test metadata to get correct IDs
        test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_meta["id"], "target": preds.flatten()})

        # Save to disk
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} does not exceed threshold {THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
