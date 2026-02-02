import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library import config
from library import data_loader
from library import expert_factory
from library import ensemble


def run_pipeline_demonstration():
    print("============================================================")
    print("   Leaf Classification Pipeline Demonstration")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Data Loading & Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Dataset and Extracting Features...")

    # This function handles:
    # - Reading metadata CSVs
    # - Extracting 'Global' features (Shape, Margin, Texture)
    # - Extracting 'Macro' features (Morphometrics from images via features.py)
    # - Caching results to parquet files in working/idea_34/
    data = data_loader.load_dataset(load_cached_data=True)

    # Validation: Check data dictionary structure and shapes
    assert "train" in data, "Data dictionary missing 'train' key"
    assert "val" in data, "Data dictionary missing 'val' key"
    assert "test" in data, "Data dictionary missing 'test' key"
    assert "classes" in data, "Data dictionary missing 'classes' key"

    n_classes = len(data["classes"])
    print(f"   - Number of classes: {n_classes}")
    print(f"   - Train samples: {len(data['train']['y'])}")
    print(f"   - Val samples:   {len(data['val']['y'])}")
    print(f"   - Test samples:  {len(data['test']['ids'])}")

    # Verify feature dimensions
    # Global features should be 192 (64*3)
    assert data["train"]["X_global"].shape[1] == 192
    # Macro features depend on features.py implementation (currently 11)
    assert data["train"]["X_macro"].shape[1] > 0

    # -------------------------------------------------------------------------
    # 2. Expert Library Construction
    # -------------------------------------------------------------------------
    print("\n[Step 2] Building Expert Library...")

    # Uses expert_factory.py to create a list of sklearn Pipelines
    # Includes Basis A (Parametric), Basis B (Non-Parametric), Basis C (Morphometric)
    experts = expert_factory.build_expert_library()

    print(f"   - Constructed {len(experts)} candidate experts.")

    # Validation: Check expert structure
    for i, exp in enumerate(experts):
        assert "id" in exp, f"Expert {i} missing 'id'"
        assert "model" in exp, f"Expert {i} missing 'model'"
        assert "view" in exp, f"Expert {i} missing 'view'"
        assert exp["view"] in ["global", "macro"], f"Invalid view for expert {i}"

    # -------------------------------------------------------------------------
    # 3. Ensemble Selection (Greedy Forward Selection)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training and Selecting Ensemble...")

    # Instantiate the GreedySelector from ensemble.py
    # We use a strict tolerance and fewer iterations for the demo to ensure speed,
    # though the dataset is small enough for full execution.
    selector = ensemble.GreedySelector(max_iterations=15, tolerance=1e-5)

    # Fit the selector:
    # 1. Trains all experts on 'train' split
    # 2. Evaluates on 'val' split
    # 3. Iteratively adds experts that minimize Log Loss
    selector.fit(experts, data)

    # Validation: Ensure at least one expert was selected
    assert (
        len(selector.selected_experts) > 0
    ), "Ensemble selection failed to select any experts."
    assert len(selector.weights) == len(selector.selected_experts), "Weights mismatch."

    print(
        f"   - Selected {len(selector.selected_experts)} experts for the final ensemble."
    )

    # -------------------------------------------------------------------------
    # 4. Refitting
    # -------------------------------------------------------------------------
    print("\n[Step 4] Refitting Selected Experts on Full Data...")

    # Retrain the selected experts on combined Train + Val data
    selector.refit(data)

    # Validation: Check if instances are fitted (simple check if they are not None)
    for item in selector.selected_experts:
        assert item["instance"] is not None, "Expert instance is None after refitting."

    # -------------------------------------------------------------------------
    # 5. Prediction & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Predictions and Submission File...")

    # Generate probability predictions for the test set
    test_probs = selector.predict(data)

    # Validation: Check prediction shape and properties
    n_test_samples = len(data["test"]["ids"])
    assert test_probs.shape == (
        n_test_samples,
        n_classes,
    ), f"Prediction shape mismatch. Expected ({n_test_samples}, {n_classes}), got {test_probs.shape}"

    # Check if probabilities sum to 1 (within floating point tolerance)
    row_sums = test_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    # Check range [0, 1]
    assert (test_probs >= 0).all() and (
        test_probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Construct DataFrame
    submission_df = pd.DataFrame(test_probs, columns=data["classes"])
    submission_df.insert(0, "id", data["test"]["ids"])

    # Save to disk
    print(f"   - Saving to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    # -------------------------------------------------------------------------
    # 6. Final Verification
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission Format...")

    if os.path.exists(config.SAMPLE_SUBMISSION_PATH):
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

        # Verify columns match
        assert list(submission_df.columns) == list(
            sample_sub.columns
        ), "Submission columns do not match sample submission."

        # Verify row count matches
        assert len(submission_df) == len(
            sample_sub
        ), f"Submission row count ({len(submission_df)}) does not match sample ({len(sample_sub)})."

        print("   - Format verification passed.")
    else:
        print("   - Sample submission not found, skipping comparison.")

    print("\nSUCCESS: Pipeline execution completed.")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(config.RANDOM_SEED)

    # Run the demo
    run_pipeline_demonstration()
