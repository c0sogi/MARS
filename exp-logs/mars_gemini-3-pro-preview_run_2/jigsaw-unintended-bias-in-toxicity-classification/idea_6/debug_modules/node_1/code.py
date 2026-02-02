import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import load_datasets
from library.model_arch import ToxicityModel
from library.engine import Engine
from library.metrics import BiasMetricCalculator


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    print("=== Setting up Configuration for Demo Run ===")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small sample for quick execution
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to use the demo directory
    Config.CACHE_TRAIN_INPUT_IDS = os.path.join(
        Config.WORKING_DIR, "train_input_ids_debug.npy"
    )
    Config.CACHE_TRAIN_ATTN_MASKS = os.path.join(
        Config.WORKING_DIR, "train_masks_debug.npy"
    )
    Config.CACHE_TRAIN_TARGETS = os.path.join(
        Config.WORKING_DIR, "train_targets_debug.npy"
    )
    Config.CACHE_TRAIN_AUX_TARGETS = os.path.join(
        Config.WORKING_DIR, "train_aux_debug.npy"
    )
    Config.CACHE_TRAIN_SAMPLE_WEIGHTS = os.path.join(
        Config.WORKING_DIR, "train_weights_debug.npy"
    )

    Config.CACHE_VAL_INPUT_IDS = os.path.join(
        Config.WORKING_DIR, "val_input_ids_debug.npy"
    )
    Config.CACHE_VAL_ATTN_MASKS = os.path.join(
        Config.WORKING_DIR, "val_masks_debug.npy"
    )
    Config.CACHE_VAL_TARGETS = os.path.join(Config.WORKING_DIR, "val_targets_debug.npy")
    Config.CACHE_VAL_AUX_TARGETS = os.path.join(Config.WORKING_DIR, "val_aux_debug.npy")
    Config.CACHE_VAL_IDS = os.path.join(Config.WORKING_DIR, "val_ids_debug.npy")

    Config.CACHE_TEST_INPUT_IDS = os.path.join(
        Config.WORKING_DIR, "test_input_ids_debug.npy"
    )
    Config.CACHE_TEST_ATTN_MASKS = os.path.join(
        Config.WORKING_DIR, "test_masks_debug.npy"
    )
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids_debug.npy")

    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n=== Loading Datasets ===")
    # Force reload to generate debug cache files
    train_dataset, val_dataset, test_dataset = load_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size:   {len(val_dataset)}")
    print(f"Test Dataset Size:  {len(test_dataset)}")

    # Verify dataset size matches debug config
    assert len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    assert len(val_dataset) == Config.DEBUG_SAMPLE_SIZE
    # Test set might be smaller if source file is small, but usually matches debug limit

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debug
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n=== Initializing Model ===")
    model = ToxicityModel(checkpoint=Config.MODEL_NAME)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Engine
    engine = Engine(model, optimizer, device)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n=== Starting Training (1 Epoch) ===")
    train_loss = engine.train_epoch(train_loader, epoch=0)
    print(f"Training completed. Loss: {train_loss:.4f}")

    # Verify loss is a valid number
    if np.isnan(train_loss) or np.isinf(train_loss):
        raise ValueError("Training loss is NaN or Inf.")

    # --------------------------------------------------------------------------
    # 5. Evaluation Loop
    # --------------------------------------------------------------------------
    print("\n=== Starting Evaluation ===")
    val_loss, metrics = engine.evaluate(val_loader)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Overall AUC:     {metrics['overall_auc']:.4f}")
    print(f"Final Score:     {metrics['final_score']:.4f}")

    # --------------------------------------------------------------------------
    # 6. Prediction Loop
    # --------------------------------------------------------------------------
    print("\n=== Generating Predictions ===")
    predictions = engine.predict(test_loader)

    print(f"Generated {len(predictions)} predictions.")

    # Create submission file
    submission_df = pd.DataFrame(
        {"id": list(predictions.keys()), "prediction": list(predictions.values())}
    )
    submission_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # --------------------------------------------------------------------------
    # 7. Logic Verification: Bias Metric Calculator
    # --------------------------------------------------------------------------
    print("\n=== Verifying Bias Metric Logic ===")
    # We create synthetic data to verify the metric calculator handles edge cases
    # and computes the generalized mean correctly.

    calculator = BiasMetricCalculator()

    # Synthetic Data
    # 4 samples, 2 identities (male, female)
    # y_true: [0, 1, 0, 1] - Mixed targets to ensure variance for AUC
    # y_pred: [0.1, 0.9, 0.2, 0.8]
    # Identities:
    #   Sample 0: Male (Non-toxic)
    #   Sample 1: Male (Toxic)
    #   Sample 2: Female (Non-toxic)
    #   Sample 3: Female (Toxic)

    y_true_syn = np.array([0, 1, 0, 1])
    y_pred_syn = np.array([0.1, 0.9, 0.2, 0.8])

    # Identity matrix: [male, female, ...others 0]
    identities_syn = np.zeros((4, len(Config.IDENTITY_COLS)))
    # Set 'male' (index 0) for samples 0, 1
    identities_syn[0, 0] = 1
    identities_syn[1, 0] = 1
    # Set 'female' (index 1) for samples 2, 3
    identities_syn[2, 1] = 1
    identities_syn[3, 1] = 1

    results, df_metrics = calculator.calculate_bias_metrics(
        y_true_syn, y_pred_syn, identities_syn
    )

    print("Synthetic Metric Results:")
    print(f"  Overall AUC: {results['overall_auc']:.4f}")
    print(f"  Mp Subgroup: {results['mp_subgroup_auc']:.4f}")
    print(f"  Mp BPSN:     {results['mp_bpsn_auc']:.4f}")
    print(f"  Mp BNSP:     {results['mp_bnsp_auc']:.4f}")

    # Assertions
    # We just assert it runs and produces a score between 0 and 1.
    if not (0.0 <= results["final_score"] <= 1.0):
        raise AssertionError(
            f"Final score {results['final_score']} out of range [0, 1]"
        )

    # Check Generalized Mean logic (p=-5)
    # If we have scores [0.5, 0.5], mean should be 0.5
    # If we have scores [0.1, 0.9], mean should be dominated by 0.1 (closer to min)
    scores = np.array([0.1, 0.9])
    p = -5
    expected_gm = np.power(np.mean(np.power(scores, p)), 1 / p)
    computed_gm = calculator._compute_generalized_mean(scores, p)

    if abs(expected_gm - computed_gm) > 1e-6:
        raise AssertionError(
            f"Generalized mean calculation failed. Expected {expected_gm}, got {computed_gm}"
        )

    print("Metric logic verification passed.")
    print("\n=== Demo Run Completed Successfully ===")


if __name__ == "__main__":
    main()
