import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_splits
from library.feature_extractor import FeatureEngineer
from library.dataset import PizzaDataset
from library.models import RandomForestModel, FiLMClassifier
from library.engine import train_mlp, evaluate_mlp, train_rf, predict_ensemble


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Load Data
    # load_cached_data=True ensures we use pre-processed parquet files if available
    print("Loading data splits...")
    train_df, val_df, test_df = get_splits(load_cached_data=True)

    # Extract labels
    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 3. Feature Engineering
    print("Generating features...")
    fe = FeatureEngineer()

    # Returns ((rf_train, rf_val, rf_test), (mlp_train, mlp_val, mlp_test))
    # rf_feats are sparse matrices, mlp_feats are dicts of tensors
    (rf_train, rf_val, rf_test), (mlp_train, mlp_val, mlp_test) = fe.fit_transform(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Prepare Datasets for MLP
    print("Preparing DataLoaders...")
    # Create Datasets
    train_dataset = PizzaDataset(mlp_train, labels=y_train)
    val_dataset = PizzaDataset(mlp_val, labels=y_val)
    # Test dataset has no labels
    test_dataset = PizzaDataset(mlp_test, labels=None)

    # Create DataLoaders
    # Shuffle train, but keep val/test ordered for alignment
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 5. Train Models

    # --- Stream A: Random Forest ---
    rf_model = RandomForestModel()
    train_rf(rf_model, rf_train, y_train)

    # --- Stream B: FiLM MLP ---
    # Determine control dimension from the first sample in the dataset
    # mlp_train is a dict of tensors. 'control_features' key holds the vector.
    control_dim = mlp_train["control_features"].shape[1]

    mlp_model = FiLMClassifier(control_input_dim=control_dim)
    mlp_model.to(device)

    # Train MLP
    mlp_model, best_mlp_auc = train_mlp(mlp_model, train_loader, val_loader, device)

    # 6. Evaluation & Ensemble
    print("Evaluating Ensemble on Validation Set...")

    # RF Predictions
    rf_probs_val = rf_model.predict_proba(rf_val)

    # MLP Predictions
    # We manually run inference to get probabilities aligned with val_df
    mlp_model.eval()
    mlp_probs_val_list = []
    with torch.no_grad():
        for batch_inputs, _ in val_loader:
            inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            logits = mlp_model(inputs)
            probs = torch.sigmoid(logits)
            mlp_probs_val_list.extend(probs.cpu().numpy())
    mlp_probs_val = np.array(mlp_probs_val_list).flatten()

    # Weighted Average Ensemble
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP
    ensemble_probs_val = (w_rf * rf_probs_val) + (w_mlp * mlp_probs_val)

    # Compute Metric
    final_auc = roc_auc_score(y_val, ensemble_probs_val)

    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error
    errors = np.abs(y_val - ensemble_probs_val)

    # Select numerical columns for correlation
    analysis_df = val_df.select_dtypes(include=[np.number]).copy()
    analysis_df["prediction_error"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["prediction_error"]
        .drop("prediction_error")
        .sort_values(ascending=False)
    )

    print(
        "Top Positive Correlations with Error (Features associated with higher error):"
    )
    print(correlations.head(5))
    print(
        "\nTop Negative Correlations with Error (Features associated with lower error):"
    )
    print(correlations.tail(5))

    # 8. Submission
    # Threshold check
    threshold = 0.7135451153926904
    if final_auc > threshold:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )
        predict_ensemble(
            rf_model=rf_model,
            mlp_model=mlp_model,
            rf_test_feats=rf_test,
            mlp_test_loader=test_loader,
            test_df=test_df,
            device=device,
        )
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
