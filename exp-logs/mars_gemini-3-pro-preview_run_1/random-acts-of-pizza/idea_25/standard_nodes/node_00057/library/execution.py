import torch
import numpy as np
from torch.utils.data import DataLoader
from library.utils import set_seed, get_device, save_submission
from library.torch_dataset import get_pizza_datasets
from library.neural_model import train_neural_model
from library.rf_model import run_rf_pipeline


def run_mlp_pipeline(
    load_cached_data=True,
    debug_mode=False,
    debug_size=50,
    epochs=50,
    patience=15,
    batch_size=32,
):
    """
    Orchestrates the Neural Network stream: Data loading, Training, and Inference.
    """
    device = get_device()
    print("\n--- Starting Neural Network Pipeline ---")

    # 1. Get Datasets
    train_ds, val_ds, test_ds = get_pizza_datasets(
        load_cached_data=load_cached_data,
        debug_mode=debug_mode,
        debug_size=debug_size,
    )

    # 2. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Determine Input Dimensions
    # Fetch a sample to check shapes
    sample_item = train_ds[0]
    input_dims = {
        "text_dim": sample_item["request_emb"].shape[0],  # Typically 384
        "meta_dim": sample_item["metadata"].shape[
            0
        ],  # Dependent on feature engineering
    }

    # 4. Define Config
    config = {
        "lr": 1e-4,
        "epochs": epochs,
        "patience": patience,
        "hidden_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
    }

    # 5. Train Model
    model, history = train_neural_model(
        train_loader, val_loader, input_dims, config=config
    )

    # 6. Inference on Test Set
    print("Generating MLP Test Predictions...")
    model.eval()
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            req = batch["request_emb"].to(device)
            hist = batch["history_seq"].to(device)
            mask = batch["history_mask"].to(device)
            meta = batch["metadata"].to(device)

            # Forward pass
            logits = model(req, hist, mask, meta)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            test_probs.extend(probs)

    test_probs = np.array(test_probs)

    # Get best validation AUC
    best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0.0

    return best_val_auc, test_probs


def evaluate_ensemble(
    load_cached_data=True,
    debug_mode=False,
    debug_size=50,
    epochs=50,
    patience=15,
):
    """
    Runs both RF and MLP pipelines, averages predictions, and saves submission.
    """
    # Set seed for reproducibility
    set_seed(42)

    # --- 1. Random Forest Stream ---
    print("\n=== Executing Random Forest Stream ===")
    # run_rf_pipeline handles data loading, training, and prediction internally.
    # We set save_output=False because we want to ensemble first.
    rf_val_auc, rf_probs, test_ids = run_rf_pipeline(
        load_cached_data=load_cached_data,
        debug_mode=debug_mode,
        debug_size=debug_size,
        save_output=False,
    )

    # --- 2. Neural Network Stream ---
    print("\n=== Executing Neural Network Stream ===")
    mlp_val_auc, mlp_probs = run_mlp_pipeline(
        load_cached_data=load_cached_data,
        debug_mode=debug_mode,
        debug_size=debug_size,
        epochs=epochs,
        patience=patience,
    )

    # --- 3. Ensemble (Simple Average) ---
    print("\n=== Computing Ensemble Predictions ===")
    # Ensure shapes match
    if len(rf_probs) != len(mlp_probs):
        raise ValueError(f"Shape mismatch: RF={len(rf_probs)}, MLP={len(mlp_probs)}")

    final_probs = 0.5 * rf_probs + 0.5 * mlp_probs

    # --- 4. Reporting ---
    print("\nFinal Validation Metrics:")
    print(f"Random Forest Best Val AUC: {rf_val_auc}")
    print(f"Neural Network Best Val AUC: {mlp_val_auc}")

    # --- 5. Save Submission ---
    save_submission(test_ids, final_probs, output_path="./submission/submission.csv")

    return {"rf_auc": rf_val_auc, "mlp_auc": mlp_val_auc}
