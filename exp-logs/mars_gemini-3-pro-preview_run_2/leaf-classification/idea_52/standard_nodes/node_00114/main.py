import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import (
    RANDOM_SEED,
    EXPERT_LIBRARY_CONFIG,
    SUBMISSION_PATH,
    FLOAT_PRECISION,
)
from library.data_manager import LeafData
from library.model_engine import GreedySelector, WeightedEnsemble
from library.utils import save_submission, clipped_log_loss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    np.random.seed(RANDOM_SEED)
    print("Starting CW-PPGE Workflow...")

    # 2. Data Loading
    print("Loading datasets...")
    data_manager = LeafData()
    # Load data (this handles caching and feature extraction)
    datasets = data_manager.load_datasets(load_cached_data=True)

    train_data = datasets["train"]
    val_data = datasets["val"]
    test_data = datasets["test"]
    class_names = datasets["classes"]

    y_train = train_data["y"]
    y_val = val_data["y"]

    print(f"Train samples: {len(y_train)}")
    print(f"Val samples: {len(y_val)}")
    print(f"Test samples: {len(test_data['ids'])}")
    print(f"Classes: {len(class_names)}")

    # 3. Phase 1: Expert Selection (Train on Train, Evaluate on Val)
    print("\n--- Phase 1: Greedy Expert Selection ---")
    selector = GreedySelector(
        expert_configs=EXPERT_LIBRARY_CONFIG, max_steps=50, tolerance=1e-6
    )

    # Fit selector
    # This trains all candidates on train_data and selects based on val_data performance
    selected_experts = selector.fit(train_data, y_train, val_data, y_val)

    final_val_metric = selector.best_loss
    # Print the metric exactly as requested
    print(f"Final Validation Metric: {final_val_metric:.16f}")

    # 4. Failure Analysis on Validation Set
    print("\n--- Failure Analysis ---")
    # Re-instantiate ensemble with selected experts and train on train_data only
    # to get predictions on val_data for analysis
    analysis_ensemble = WeightedEnsemble(selected_experts)
    analysis_ensemble.fit(train_data, y_train)
    val_probs = analysis_ensemble.predict_proba(val_data)

    # Calculate per-sample log loss
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Gather true class probabilities
    rows = np.arange(len(y_val))
    true_class_probs = val_probs_clipped[rows, y_val]
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Sample Loss: {np.mean(sample_losses):.6f}")

    # Correlate with Morphometric Features
    # Morphometrics: [Hu(7), Aspect, Solidity, Extent, Eccentricity]
    morph_names = [f"Hu_{i}" for i in range(7)] + [
        "AspectRatio",
        "Solidity",
        "Extent",
        "Eccentricity",
    ]
    X_morph = val_data["morphometrics"]

    print("Correlation between Error (Log Loss) and Morphometric Features:")
    for i, name in enumerate(morph_names):
        feat_values = X_morph[:, i]
        # Handle constant features to avoid warnings
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(sample_losses, feat_values)
        print(f"  {name}: {corr:.4f}")

    # 5. Phase 2: Final Retraining (Full Data)
    print("\n--- Phase 2: Final Retraining ---")

    # Combine Train and Val
    full_data = {}
    keys_to_concat = ["global", "margin", "shape", "texture", "morphometrics", "ids"]
    for key in keys_to_concat:
        full_data[key] = np.concatenate([train_data[key], val_data[key]], axis=0)

    y_full = np.concatenate([y_train, y_val], axis=0)

    print(f"Full training set size: {len(y_full)}")

    # Initialize final ensemble
    final_ensemble = WeightedEnsemble(selected_experts)
    final_ensemble.fit(full_data, y_full)

    # 6. Submission
    print("\n--- Generating Submission ---")

    # Predict on Test
    test_probs = final_ensemble.predict_proba(test_data)

    # Save submission
    # We use a safe threshold to ensure submission is generated
    threshold = 10.0
    if final_val_metric < threshold:
        save_submission(
            ids=test_data["ids"],
            probabilities=test_probs,
            class_names=class_names,
            output_path=SUBMISSION_PATH,
        )
    else:
        print(f"Validation metric {final_val_metric} too high. Submission skipped.")


if __name__ == "__main__":
    main()
