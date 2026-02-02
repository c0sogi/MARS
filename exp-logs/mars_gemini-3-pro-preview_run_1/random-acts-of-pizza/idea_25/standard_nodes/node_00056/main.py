import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.utils import set_seed, get_device, save_submission
from library.rf_model import RFPipeline
from library.torch_dataset import get_pizza_datasets
from library.neural_model import train_neural_model


def main():
    # 1. Configuration
    set_seed(42)
    device = get_device()

    # Parameters
    # Using full data (DEBUG_MODE=False) to ensure high performance.
    # The dataset size (approx 2.3k) allows for very fast training even with full data.
    LOAD_CACHED = True
    DEBUG_MODE = False
    DEBUG_SIZE = 50
    EPOCHS = 30  # Sufficient for convergence on this dataset size
    PATIENCE = 10
    BATCH_SIZE = 32
    THRESHOLD = 0.6959737721862433

    print("=== Starting Orchestration ===")

    # ---------------------------------------------------------
    # 2. Random Forest Stream
    # ---------------------------------------------------------
    print("\n--- Random Forest Stream ---")
    rf_pipeline = RFPipeline(n_estimators=500, min_samples_leaf=2, random_state=42)

    # Get Data
    rf_data = rf_pipeline.get_data(
        load_cached_data=LOAD_CACHED, debug_mode=DEBUG_MODE, debug_size=DEBUG_SIZE
    )

    # Train
    rf_pipeline.train(
        rf_data["X_train"], rf_data["y_train"], rf_data["X_val"], rf_data["y_val"]
    )

    # Inference
    rf_val_probs = rf_pipeline.predict(rf_data["X_val"])
    rf_test_probs = rf_pipeline.predict(rf_data["X_test"])
    test_ids = rf_data["test_ids"]

    # ---------------------------------------------------------
    # 3. Neural Network Stream
    # ---------------------------------------------------------
    print("\n--- Neural Network Stream ---")

    # Get Datasets
    train_ds, val_ds, test_ds = get_pizza_datasets(
        load_cached_data=LOAD_CACHED,
        debug_mode=DEBUG_MODE,
        debug_size=DEBUG_SIZE,
    )

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Determine Input Dimensions from a sample
    sample_item = train_ds[0]
    input_dims = {
        "text_dim": sample_item["request_emb"].shape[0],
        "meta_dim": sample_item["metadata"].shape[0],
    }

    # Model Config
    mlp_config = {
        "lr": 1e-4,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "hidden_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
    }

    # Train
    mlp_model, mlp_history = train_neural_model(
        train_loader, val_loader, input_dims, config=mlp_config
    )

    # Inference (Val)
    mlp_model.eval()
    mlp_val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            req = batch["request_emb"].to(device)
            hist = batch["history_seq"].to(device)
            mask = batch["history_mask"].to(device)
            meta = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            logits = mlp_model(req, hist, mask, meta)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            mlp_val_probs.extend(probs)
            val_targets.extend(labels.cpu().numpy())

    mlp_val_probs = np.array(mlp_val_probs)
    val_targets = np.array(val_targets)

    # Inference (Test)
    mlp_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            req = batch["request_emb"].to(device)
            hist = batch["history_seq"].to(device)
            mask = batch["history_mask"].to(device)
            meta = batch["metadata"].to(device)

            logits = mlp_model(req, hist, mask, meta)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            mlp_test_probs.extend(probs)

    mlp_test_probs = np.array(mlp_test_probs)

    # ---------------------------------------------------------
    # 4. Ensemble & Validation
    # ---------------------------------------------------------
    print("\n--- Ensemble & Validation ---")

    # Simple Weighted Average Ensemble
    ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * mlp_val_probs
    ensemble_test_probs = 0.5 * rf_test_probs + 0.5 * mlp_test_probs

    # Calculate Final Metric
    final_auc = roc_auc_score(val_targets, ensemble_val_probs)

    print(f"Final Validation Metric: {final_auc}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Load interpretable metadata for validation set
    df_val = pd.read_csv("./metadata/val.csv")

    # Calculate Error Magnitude
    errors = np.abs(val_targets - ensemble_val_probs)

    # Select numeric features for correlation
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [
        "requester_received_pizza",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    correlations = {}
    for col in numeric_cols:
        # Handle potential NaNs in raw metadata
        feat_values = df_val[col].fillna(0).values
        if len(feat_values) == len(errors):
            # Compute correlation
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort and Print Top Correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top Correlations with Error Magnitude:")
    for name, val in sorted_corr[:10]:
        print(f"{name}: {val:.4f}")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Saving submission."
        )
        save_submission(
            test_ids, ensemble_test_probs, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"\nValidation AUC ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
