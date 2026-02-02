import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import get_dataloaders, get_test_dataloader, get_tfidf_features
from library.models import CustomTransformer, LinearModelWrapper
from library.trainer import run_transformer_training
from library.ensemble import Ensemble


def main():
    # ====================================================
    # 1. Setup and Configuration Overrides for Demo
    # ====================================================
    print("[1/6] Setting up configuration for fast demonstration...")

    # Override Config for speed
    Config.debug = True
    Config.epochs = 1
    Config.train_batch_size = 8
    Config.valid_batch_size = 16
    Config.working_dir = "./working/demo_check"
    os.makedirs(Config.working_dir, exist_ok=True)

    # Use a tiny model for demonstration purposes to avoid long download/train times
    # 'prajjwal1/bert-tiny' is extremely small (L=2, H=128)
    demo_model_name = "prajjwal1/bert-tiny"

    seed_everything(Config.seed)
    print("Configuration set. Debug mode enabled.")

    # ====================================================
    # 2. Data Loader Verification
    # ====================================================
    print("\n[2/6] Verifying Data Loaders...")

    # We need a tokenizer for the data loader. We use the one corresponding to our demo model.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(demo_model_name)

    # Test Train/Val Loaders
    train_loader, val_loader = get_dataloaders(tokenizer)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Assertions
    assert "ids" in batch, "Batch missing 'ids'"
    assert "mask" in batch, "Batch missing 'mask'"
    assert "targets" in batch, "Batch missing 'targets'"

    # Check shapes
    # Batch size is Config.train_batch_size (8), Sequence length is Config.max_len (200)
    assert batch["ids"].shape == (
        Config.train_batch_size,
        Config.max_len,
    ), f"Incorrect input shape: {batch['ids'].shape}"
    assert batch["targets"].shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), f"Incorrect target shape: {batch['targets'].shape}"

    print(f"Data Loader check passed. Batch shape: {batch['ids'].shape}")

    # ====================================================
    # 3. Transformer Model Verification (Training Loop)
    # ====================================================
    print("\n[3/6] Verifying Transformer Training Loop (TinyBERT)...")

    # We run the training function provided in library.trainer
    # This tests Model instantiation, LLRD optimizer setup, Forward pass, Backward pass, and Validation
    best_auc = run_transformer_training(
        model_name=demo_model_name, save_name="tiny_bert_demo.bin"
    )

    # Check if model file was created
    model_path = os.path.join(Config.working_dir, "tiny_bert_demo.bin")
    assert os.path.exists(model_path), "Model checkpoint was not saved."
    assert 0 <= best_auc <= 1, f"Invalid AUC score: {best_auc}"

    print(f"Transformer training check passed. Best AUC: {best_auc:.4f}")

    # ====================================================
    # 4. Linear Model Verification (TF-IDF)
    # ====================================================
    print("\n[4/6] Verifying Linear Model (TF-IDF)...")

    # Generate features (Debug mode ensures this is fast)
    # Force load_cached_data=False to ensure we test the generation logic
    X_train, X_val, X_test, y_train, y_val = get_tfidf_features(load_cached_data=False)

    # Check shapes
    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Mismatch in Train features and labels count"
    assert X_val.shape[0] == y_val.shape[0], "Mismatch in Val features and labels count"

    # Initialize and Fit Linear Model
    linear_model = LinearModelWrapper()
    linear_model.fit(X_train, y_train)

    # Predict
    val_preds_linear = linear_model.predict_proba(X_val)

    # Verify predictions
    assert val_preds_linear.shape == y_val.shape, "Prediction shape mismatch"
    linear_auc = compute_auc(y_val, val_preds_linear)
    print(f"Linear Model check passed. Val AUC: {linear_auc:.4f}")

    # ====================================================
    # 5. Ensemble Verification
    # ====================================================
    print("\n[5/6] Verifying Ensemble Optimization...")

    # Create synthetic predictions to simulate a second model
    # (In a real scenario, this would be the Transformer's predictions)
    # We add some noise to the ground truth to simulate a decent model
    rng = np.random.default_rng(Config.seed)
    noise = rng.uniform(0, 0.4, size=y_val.shape)
    # Simple synthetic prediction: (0.7 * truth + 0.3 * noise) clipped
    val_preds_synthetic = np.clip(0.6 * y_val + 0.4 * noise, 0, 1)

    # List of predictions from different models
    preds_list = [val_preds_linear, val_preds_synthetic]

    ensemble = Ensemble()

    # Optimize weights
    weights = ensemble.optimize_weights(preds_list, y_val)

    # Validation
    assert len(weights) == 2, "Should have 2 weights for 2 models"
    assert np.isclose(sum(weights), 1.0), f"Weights must sum to 1, got {sum(weights)}"

    # Blend
    blended_preds = ensemble.blend_predictions(preds_list)
    final_auc = compute_auc(y_val, blended_preds)

    print(
        f"Ensemble check passed. Optimized Weights: {weights}. Final Blended AUC: {final_auc:.4f}"
    )

    # ====================================================
    # 6. Metric Verification
    # ====================================================
    print("\n[6/6] Verifying Metric Calculation...")

    # Test compute_auc with known values
    y_true_test = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    # Perfect predictions
    y_pred_perfect = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.9, 0.1]])
    # Worst predictions
    y_pred_worst = 1 - y_pred_perfect

    score_perfect = compute_auc(y_true_test, y_pred_perfect)
    score_worst = compute_auc(y_true_test, y_pred_worst)

    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect score should be 1.0, got {score_perfect}"
    assert score_worst < 0.5, f"Worst score should be < 0.5, got {score_worst}"

    print("Metric check passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
