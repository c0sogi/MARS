import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
from library.features import FeatureProcessor
from library.model_rf import RFWrapper
from library.model_mlp import MLPTrainer, PizzaDataset, set_seed

# ==========================================
# Configuration Override for Fast Baseline
# ==========================================
# Limit training steps to ensure execution within time limits
config.EPOCHS = 10
config.RF_ESTIMATORS = 100
config.RF_N_JOBS = -1  # Use all cores

# Ensure reproducibility
set_seed(config.RANDOM_STATE)


def run():
    print("Starting execution of runfile.py...")

    # ==========================================
    # 1. Feature Processing
    # ==========================================
    print("Processing features...")
    processor = FeatureProcessor()
    # Load cached data if available to save time
    train_data, val_data, test_data = processor.process(load_cached_data=True)

    # ==========================================
    # 2. Random Forest Stream
    # ==========================================
    print("Running Random Forest Stream...")
    rf_model = RFWrapper()

    # Train RF
    rf_model.train(train_data["rf_features"], train_data["labels"])

    # Predict RF
    rf_val_preds = rf_model.predict(val_data["rf_features"])
    rf_test_preds = rf_model.predict(test_data["rf_features"])

    # ==========================================
    # 3. MLP Stream
    # ==========================================
    print("Running MLP Stream...")

    # Prepare DataLoaders
    # Using the PizzaDataset class from library.model_mlp
    train_dataset = PizzaDataset(train_data, mode="train")
    val_dataset = PizzaDataset(val_data, mode="val")
    test_dataset = PizzaDataset(test_data, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Metadata dimension for the MLP input
    metadata_dim = train_data["mlp_metadata"].shape[1]
    mlp_trainer = MLPTrainer(metadata_dim, device)

    # Train MLP
    mlp_trainer.train(train_loader, val_loader)

    # Predict MLP
    mlp_val_preds = mlp_trainer.predict(val_loader)
    mlp_test_preds = mlp_trainer.predict(test_loader)

    # ==========================================
    # 4. Ensemble
    # ==========================================
    print("Ensembling predictions...")
    # Simple weighted average
    val_preds = (config.ENSEMBLE_WEIGHT_RF * rf_val_preds) + (
        config.ENSEMBLE_WEIGHT_MLP * mlp_val_preds
    )
    test_preds = (config.ENSEMBLE_WEIGHT_RF * rf_test_preds) + (
        config.ENSEMBLE_WEIGHT_MLP * mlp_test_preds
    )

    # ==========================================
    # 5. Evaluation
    # ==========================================
    val_labels = val_data["labels"]
    auc = roc_auc_score(val_labels, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    # Load raw validation metadata for interpretable analysis
    df_val = pd.read_csv(config.VAL_PATH)

    # Calculate residuals (absolute error)
    errors = np.abs(val_labels - val_preds)

    # Calculate correlation between numeric features and error
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        # Skip target and ID columns if present as numeric
        if col not in [
            "requester_received_pizza",
            "unix_timestamp_of_request",
            "unix_timestamp_of_request_utc",
        ]:
            # Handle potential NaNs in raw data
            series = df_val[col].fillna(0)
            if series.std() > 0:  # Avoid constant columns
                corr = np.corrcoef(series, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    print("Top 10 Features correlated with Error Magnitude:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, score in sorted_corr[:10]:
        print(f"{name}: {score:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.6959737721862433

    if auc > THRESHOLD:
        print(f"\nValidation metric {auc} > {THRESHOLD}. Generating submission...")

        # Load test metadata to get request_ids
        df_test = pd.read_csv(config.TEST_PATH)

        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": test_preds,
            }
        )

        # Ensure submission directory exists
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nValidation metric {auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
