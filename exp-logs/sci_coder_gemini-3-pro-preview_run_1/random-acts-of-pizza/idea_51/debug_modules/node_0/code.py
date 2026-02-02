import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import MLP_PARAMS, RF_PARAMS, WORKING_DIR, TEST_CSV
from library.utils import seed_everything, save_submission, load_from_cache
from library.feature_engineering import run_feature_engineering
from library.rf_learner import train_rf, predict_rf, extract_sparse
from library.dataset import get_dataloaders
from library.engine import train_mlp_model, predict_mlp, evaluate_models


def main():
    print("Starting demonstration of the Pizza Prediction Pipeline...")

    # 1. Setup and Configuration for Speed
    # We override the configuration dictionaries in memory to speed up the demo
    seed_everything(42)

    print("Adjusting hyperparameters for fast demonstration...")
    MLP_PARAMS["max_epochs"] = 2  # Reduce from 50 to 2
    MLP_PARAMS["patience"] = 1  # Reduce patience
    RF_PARAMS["n_estimators"] = 10  # Reduce from 500 to 10
    RF_PARAMS["n_jobs"] = 1  # Avoid overhead in demo

    # 2. Feature Engineering
    # We run this with load_cached_data=False to demonstrate the generation logic.
    # This handles text processing (SBERT) and tabular feature creation.
    print("\n=== Step 1: Feature Engineering ===")
    rf_data, mlp_data = run_feature_engineering(load_cached_data=False)

    # Validate outputs
    assert "X_train" in rf_data, "RF data missing X_train"
    assert "X_train" in mlp_data, "MLP data missing X_train"
    print("Feature engineering completed successfully.")

    # 3. Stream A: Random Forest
    print("\n=== Step 2: Random Forest Stream ===")

    # Extract data (handling potential sparse matrix wrapping)
    X_train_rf = extract_sparse(rf_data["X_train"])
    y_train_rf = extract_sparse(rf_data["y_train"])
    X_val_rf = extract_sparse(rf_data["X_val"])
    y_val_rf = extract_sparse(rf_data["y_val"])
    X_test_rf = extract_sparse(rf_data["X_test"])

    # Train RF
    print(f"Training Random Forest on {X_train_rf.shape[0]} samples...")
    rf_model = train_rf(X_train_rf, y_train_rf)

    # Evaluate RF
    rf_val_preds = predict_rf(rf_model, X_val_rf)
    rf_auc = roc_auc_score(y_val_rf, rf_val_preds)
    print(f"Random Forest Validation AUC: {rf_auc:.4f}")

    # Basic assertion to ensure predictions are valid probabilities
    assert np.all(
        (rf_val_preds >= 0) & (rf_val_preds <= 1)
    ), "RF predictions out of bounds"

    # 4. Stream B: MLP (PizzaNet)
    print("\n=== Step 3: MLP Stream (PizzaNet) ===")

    # Create DataLoaders
    # Note: get_dataloaders loads from the cache files created by run_feature_engineering
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # Determine input dimension for the metadata branch of the MLP
    # The dataset returns a dictionary; we check the shape of 'metadata'
    sample_batch = next(iter(train_loader))
    input_metadata_dim = sample_batch["metadata"].shape[1]
    print(f"MLP Metadata Input Dimension: {input_metadata_dim}")

    # Train MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on {device}...")
    mlp_model = train_mlp_model(
        train_loader, val_loader, input_metadata_dim, device=device
    )

    # Evaluate MLP
    mlp_val_preds = predict_mlp(mlp_model, val_loader, device=device)

    # Targets for validation are in the dataset
    y_val_mlp = mlp_data["y_val"]
    mlp_auc = roc_auc_score(y_val_mlp, mlp_val_preds)
    print(f"MLP Validation AUC: {mlp_auc:.4f}")

    assert len(mlp_val_preds) == len(y_val_mlp), "MLP prediction length mismatch"

    # 5. Ensemble and Submission
    print("\n=== Step 4: Ensemble & Submission ===")

    # Generate final test predictions using the ensemble function
    # This generates predictions for both models on the test set and averages them
    test_preds = evaluate_models(
        rf_model, mlp_model, test_loader, X_test_rf, device=device
    )

    print(f"Generated {len(test_preds)} test predictions.")

    # Load Test IDs to format submission
    if os.path.exists(TEST_CSV):
        df_test = pd.read_csv(TEST_CSV)
        request_ids = df_test["request_id"].values

        # Verify alignment
        if len(request_ids) != len(test_preds):
            raise ValueError(
                f"Mismatch: {len(request_ids)} IDs vs {len(test_preds)} preds"
            )

        # Save Submission
        save_submission(request_ids, test_preds, filename="demo_submission.csv")
    else:
        print(f"Warning: {TEST_CSV} not found. Skipping CSV generation.")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
