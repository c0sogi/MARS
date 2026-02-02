import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.dataset import get_notebook_data, generate_training_pairs
from library.fine_tune import fine_tune_models
from library.feature_engineering import generate_features
from library.regressor import train_lgbm, generate_submission


def run_demo():
    print("Initializing Demo Pipeline...")

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    # We monkey-patch the Config class to use a tiny model and minimal data
    # to ensure this demo runs within the time limit.

    print("Patching Config for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15  # Use only 15 notebooks
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4

    # Use a tiny BERT model for demonstration speed
    TINY_MODEL = "prajjwal1/bert-tiny"
    Config.MODEL_NAMES = [TINY_MODEL]

    # Update paths since MODEL_NAMES changed
    Config.MODEL_SAVE_PATHS = {
        TINY_MODEL: os.path.join(
            Config.WORKING_DIR, f"{TINY_MODEL.replace('/', '_')}_finetuned"
        )
    }

    # LightGBM fast settings
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["verbose"] = -1

    # Clean up previous working dir if exists to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)

    # =========================================================================
    # 2. Verify Utility Logic (Kendall Tau)
    # =========================================================================
    print("\n--- Verifying Kendall Tau Metric Logic ---")
    # Case 1: Perfect match
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    score_perfect = compute_kendall_tau(df_pred_perfect, df_gt)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case 2: Complete reversal (worst case)
    # For n=3, swaps=3 (c b a -> b c a -> b a c -> a b c is not bubble sort,
    # inversions: (c,b), (c,a), (b,a) = 3). Total pairs = 3*2 = 6.
    # Formula: 1 - 4 * (3 / 6) = 1 - 2 = -1.0
    df_pred_reverse = pd.DataFrame({"id": ["nb1"], "cell_order": ["c b a"]})
    score_reverse = compute_kendall_tau(df_pred_reverse, df_gt)
    assert np.isclose(score_reverse, -1.0), f"Expected -1.0, got {score_reverse}"

    print("Kendall Tau metric verification passed.")

    # =========================================================================
    # 3. Verify Data Loading
    # =========================================================================
    print("\n--- Verifying Data Loading ---")
    # Load a sample notebook from the training set
    sample_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(1)
    sample_file_path = os.path.join(
        Config.INPUT_DIR, sample_train_df.iloc[0]["file_path"]
    )

    code_cells, md_cells = get_notebook_data(sample_file_path)

    assert isinstance(code_cells, list), "code_cells should be a list"
    assert isinstance(md_cells, list), "md_cells should be a list"
    if code_cells:
        assert (
            "id" in code_cells[0] and "source" in code_cells[0]
        ), "Malformed code cell"

    print(
        f"Successfully loaded notebook: {len(code_cells)} code cells, {len(md_cells)} markdown cells."
    )

    # =========================================================================
    # 4. Fine-Tuning Step
    # =========================================================================
    print("\n--- Running Fine-Tuning (Mock) ---")
    # This will generate training pairs and fine-tune the tiny model
    # We verify that the model directory is created
    fine_tune_models(
        load_cached_data=False, epochs=Config.EPOCHS, batch_size=Config.TRAIN_BATCH_SIZE
    )

    model_save_path = Config.MODEL_SAVE_PATHS[TINY_MODEL]
    assert os.path.exists(
        model_save_path
    ), f"Model save path {model_save_path} not found after fine-tuning."
    print("Fine-tuning completed successfully.")

    # =========================================================================
    # 5. Feature Engineering Step
    # =========================================================================
    print("\n--- Running Feature Engineering ---")

    # Generate features for Train
    print("Generating Train Features...")
    train_features = generate_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_PATH,
        load_cached_data=False,
        debug=True,
    )
    assert not train_features.empty, "Train features DataFrame is empty"
    assert "target" in train_features.columns, "Target column missing in train features"

    # Generate features for Validation
    print("Generating Validation Features...")
    val_features = generate_features(
        Config.VAL_METADATA_PATH,
        Config.VAL_FEATURES_PATH,
        load_cached_data=False,
        debug=True,
    )
    assert not val_features.empty, "Val features DataFrame is empty"

    # Generate features for Test
    print("Generating Test Features...")
    test_features = generate_features(
        Config.TEST_METADATA_PATH,
        Config.TEST_FEATURES_PATH,
        load_cached_data=False,
        debug=True,
    )
    # Test features might be empty if the sampled debug notebooks have no markdown,
    # but with 15 samples it's unlikely. If empty, columns should still be correct.
    if not test_features.empty:
        assert (
            "target" not in test_features.columns
        ), "Target column should not be in test features"
        # Check for generated feature columns (using the tiny model short name)
        short_name = TINY_MODEL.split("/")[-1].replace("-", "_")
        assert any(
            c.startswith(short_name) for c in test_features.columns
        ), "Model features missing"

    print("Feature engineering completed.")

    # =========================================================================
    # 6. Regression Training Step
    # =========================================================================
    print("\n--- Training Regressor ---")
    model = train_lgbm(train_features, val_features)
    assert model is not None, "LightGBM model training failed"
    print("Regressor trained successfully.")

    # =========================================================================
    # 7. Submission Generation Step
    # =========================================================================
    print("\n--- Generating Submission ---")
    submission_df = generate_submission(model, test_features, Config.TEST_METADATA_PATH)

    # Verify submission format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    assert list(submission_df.columns) == [
        "id",
        "cell_order",
    ], "Submission columns incorrect"
    assert len(submission_df) > 0, "Submission DataFrame is empty"

    # Check content of one row
    sample_row = submission_df.iloc[0]
    cell_order = sample_row["cell_order"]
    assert isinstance(cell_order, str), "cell_order must be a string"
    assert len(cell_order.split()) > 0, "cell_order must contain cell IDs"

    print(f"Submission generated with {len(submission_df)} rows.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
