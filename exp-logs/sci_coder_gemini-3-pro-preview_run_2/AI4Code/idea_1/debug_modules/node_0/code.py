import os
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import set_seed, SUBMISSION_PATH
from library.data_loader import get_regression_data, get_inference_data
from library.model import RankPredictor, predict_notebook_order
from library.metrics import calculate_kendall_tau
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Setting Random Seed...")
    set_seed(42)
    print("    Seed set to 42.")

    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loader...")

    # Load a small subset of training data for regression (Markdown -> Rank)
    # We disable cache loading to ensure we test the raw processing logic
    df_train_subset = get_regression_data(
        data_type="train", max_samples=100, load_cached_data=False
    )

    # Verify DataFrame structure
    expected_cols = {"id", "cell_id", "text", "rank"}
    assert isinstance(
        df_train_subset, pd.DataFrame
    ), "get_regression_data should return a DataFrame"
    assert not df_train_subset.empty, "Returned DataFrame should not be empty"
    assert expected_cols.issubset(
        df_train_subset.columns
    ), f"Missing columns. Expected {expected_cols}"

    print(f"    Loaded {len(df_train_subset)} markdown cells for training.")

    # Load a small subset of validation notebooks for inference structure
    val_notebooks = get_inference_data(data_type="val", max_samples=5)

    assert isinstance(val_notebooks, list), "get_inference_data should return a list"
    assert len(val_notebooks) > 0, "Should have loaded at least one notebook"
    sample_nb = val_notebooks[0]
    assert "code_cells" in sample_nb, "Notebook dict missing 'code_cells'"
    assert "markdown_cells" in sample_nb, "Notebook dict missing 'markdown_cells'"

    print(f"    Loaded {len(val_notebooks)} validation notebooks.")
    print("    Data Loader tests passed.")

    # 3. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Training...")

    model = RankPredictor()

    # Fit on the subset loaded earlier
    model.fit(df_train_subset)

    # Verify the model can predict
    dummy_texts = ["This is an intro", "import numpy as np", "Conclusion"]
    preds = model.predict(dummy_texts)

    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert len(preds) == 3, "Should return one prediction per input text"
    assert (
        preds.dtype == np.float64 or preds.dtype == np.float32
    ), "Predictions should be floats"

    print("    Model training and prediction tests passed.")

    # 4. Prediction Logic Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Notebook Order Prediction...")

    # Use the sample notebook loaded in step 2
    # sample_nb has 'code_cells' (list of IDs) and 'markdown_cells' (list of (ID, text))
    predicted_order_str = predict_notebook_order(model, sample_nb)

    assert isinstance(predicted_order_str, str), "Prediction should be a string"
    predicted_list = predicted_order_str.split()

    # Calculate expected total cells
    total_cells = len(sample_nb["code_cells"]) + len(sample_nb["markdown_cells"])

    assert (
        len(predicted_list) == total_cells
    ), f"Predicted order length ({len(predicted_list)}) does not match total cells ({total_cells})"

    # Verify uniqueness (it's a permutation)
    assert (
        len(set(predicted_list)) == total_cells
    ), "Predicted order contains duplicate cell IDs"

    print(f"    Successfully predicted order for notebook {sample_nb['id']}.")
    print("    Prediction logic tests passed.")

    # 5. Metric Calculation Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Testing Kendall Tau Metric...")

    # Case A: Perfect Match
    # Notebook with 3 cells: A, B, C
    gt_perfect = {"nb_1": ["A", "B", "C"]}
    pred_perfect = {"nb_1": ["A", "B", "C"]}
    score_perfect = calculate_kendall_tau(pred_perfect, gt_perfect)

    # Kendall Tau should be 1.0
    assert (
        abs(score_perfect - 1.0) < 1e-6
    ), f"Perfect match should be 1.0, got {score_perfect}"

    # Case B: Complete Inversion
    # Notebook with 3 cells: A, B, C -> Predicted: C, B, A
    # n=3. Max swaps = n(n-1)/2 = 3.
    # Inversions in C, B, A relative to A, B, C: (C,B), (C,A), (B,A) -> 3 swaps.
    # Score = 1 - 4 * (Swaps / (n*(n-1))) = 1 - 4 * (3 / 6) = 1 - 2 = -1.0
    gt_reverse = {"nb_2": ["A", "B", "C"]}
    pred_reverse = {"nb_2": ["C", "B", "A"]}
    score_reverse = calculate_kendall_tau(pred_reverse, gt_reverse)

    assert (
        abs(score_reverse - (-1.0)) < 1e-6
    ), f"Reverse order should be -1.0, got {score_reverse}"

    print("    Metric calculation tests passed.")

    # 6. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[6] Running Full Inference Pipeline (Fast Mode)...")

    # Run the provided inference pipeline with very small limits to ensure speed
    # This will train, validate, and generate submission.csv
    run_inference(max_train_samples=200, max_test_samples=20)

    # Verify submission file creation
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission file missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"    Pipeline completed. Submission saved to {SUBMISSION_PATH}")
    print("    Full pipeline tests passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
