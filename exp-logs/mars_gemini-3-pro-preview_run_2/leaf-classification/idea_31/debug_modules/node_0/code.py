import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library modules are importable
sys.path.append(os.getcwd())

from library import config
from library.data_processing import DataProcessor
from library.model_library import get_expert_library
from library.ensemble_selection import GreedySelector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline_demo():
    print("=== Starting Leaf Classification Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Data Processing
    # -------------------------------------------------------------------------
    print("\n[1/4] Processing Data...")
    # Initialize DataProcessor. It handles:
    # - Loading metadata from ./metadata/
    # - Extracting macro features (Hu moments, etc.) from ./input/images/
    # - Caching results to ./working/
    # - Applying PowerTransformer (Yeo-Johnson)
    processor = DataProcessor(load_cached_data=True)
    data = processor.get_data()

    # Verify data integrity
    required_keys = [
        "X_train_combined",
        "X_train_global",
        "X_train_macro",
        "y_train",
        "X_val_combined",
        "X_val_global",
        "X_val_macro",
        "y_val",
        "X_test_combined",
        "X_test_global",
        "X_test_macro",
        "test_ids",
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in processed data dictionary."
        assert len(data[key]) > 0, f"Data for '{key}' is empty."

    print(f"Data successfully processed.")
    print(f"Training Samples: {len(data['y_train'])}")
    print(f"Validation Samples: {len(data['y_val'])}")
    print(f"Test Samples: {len(data['test_ids'])}")

    # -------------------------------------------------------------------------
    # 2. Train Experts
    # -------------------------------------------------------------------------
    print("\n[2/4] Training Probabilistic Experts...")

    # Retrieve the predefined library of experts
    experts = get_expert_library()
    print(f"Initialized {len(experts)} experts.")

    # Dictionary to map expert view requirements to data keys
    view_mapping = {
        "global": ("X_train_global", "X_val_global", "X_test_global"),
        "macro": ("X_train_macro", "X_val_macro", "X_test_macro"),
        "combined": ("X_train_combined", "X_val_combined", "X_test_combined"),
    }

    val_predictions = {}
    test_predictions = {}

    for i, expert in enumerate(experts):
        # Select appropriate data view
        train_key, val_key, test_key = view_mapping[expert.view_type]

        X_train = data[train_key]
        y_train = data["y_train"]
        X_val = data[val_key]
        X_test = data[test_key]

        # Fit the expert
        expert.fit(X_train, y_train)

        # Generate probabilities
        p_val = expert.predict_proba(X_val)
        p_test = expert.predict_proba(X_test)

        # Validate probability shapes
        n_classes = len(np.unique(y_train))
        assert p_val.shape == (len(y_train), n_classes) or p_val.shape == (
            len(data["y_val"]),
            n_classes,
        )

        # Store predictions
        val_predictions[expert.name] = p_val
        test_predictions[expert.name] = p_test

    print("All experts trained and predictions generated.")

    # -------------------------------------------------------------------------
    # 3. Ensemble Selection
    # -------------------------------------------------------------------------
    print("\n[3/4] Optimizing Ensemble (Greedy Forward Selection)...")

    # Initialize selector with a limit on ensemble size for the demo
    selector = GreedySelector(max_size=15)

    # Fit selector on validation data to find optimal combination
    selector.fit(val_predictions, data["y_val"])

    # Verify selection
    if not selector.selected_experts:
        raise RuntimeError("GreedySelector failed to select any experts.")

    print(f"Best Validation Log Loss: {selector.best_score:.5f}")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Submission...")

    # Compute weighted average of test predictions using selected experts
    final_probs = selector.predict(test_predictions)

    # Prepare DataFrame
    # Note: Classes are sorted alphabetically by sklearn, which matches sample_submission format
    classes = np.unique(data["y_train"])
    submission_df = pd.DataFrame(final_probs, columns=classes)

    # Insert ID column
    submission_df.insert(0, "id", data["test_ids"])

    # Load sample submission to verify structure
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Verify columns match
    assert list(submission_df.columns) == list(
        sample_sub.columns
    ), "Submission column mismatch."
    assert len(submission_df) == len(sample_sub), "Submission row count mismatch."

    # Save submission
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print("=== Demo Complete ===")


if __name__ == "__main__":
    # Set fixed seeds for reproducibility
    np.random.seed(config.RANDOM_STATE)
    run_pipeline_demo()
