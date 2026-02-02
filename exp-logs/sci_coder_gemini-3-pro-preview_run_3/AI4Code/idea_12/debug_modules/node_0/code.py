import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
import library.data_utils as data_utils
import library.pair_generation as pair_gen
import library.fine_tuning as fine_tuning
import library.feature_extraction as feature_ext
import library.ranker as ranker_lib
import library.inference as inference_lib


def main():
    # 1. Configuration Overrides for Demo
    # We modify the Config class attributes at runtime to create a "mini" version of the task
    print("Setting up demo configuration...")

    # Use a specific demo directory to avoid conflicts
    Config.Paths.WORKING_DIR = "./working/demo_run"
    Config.Paths.CACHE_DIR = os.path.join(Config.Paths.WORKING_DIR, "cache")
    Config.Paths.MODEL_OUTPUT_DIR = os.path.join(
        Config.Paths.WORKING_DIR, "fine_tuned_mpnet"
    )
    Config.Paths.SUBMISSION_DIR = os.path.join(Config.Paths.WORKING_DIR, "submission")
    Config.Paths.SUBMISSION_PATH = os.path.join(
        Config.Paths.SUBMISSION_DIR, "submission.csv"
    )

    # Re-create directories for the new paths
    Config.Paths.setup_dirs()

    # Reduce dataset sizes and training parameters for speed
    Config.Training.NUM_NOTEBOOKS_FINE_TUNE = 20
    Config.Training.FINE_TUNE_EPOCHS = 1
    Config.Training.FINE_TUNE_BATCH_SIZE = 4
    Config.Training.NUM_NOTEBOOKS_LGBM = 20
    Config.Training.LGBM_PARAMS["n_estimators"] = 10
    Config.Training.LGBM_PARAMS["verbose"] = -1

    # 2. Mocking get_metadata
    # The library functions are designed to load the full dataset.
    # To ensure the demo runs in < 5 minutes, we intercept the data loading
    # and return a small sample for all downstream tasks.

    original_get_metadata = data_utils.get_metadata

    def mock_get_metadata(split):
        df = original_get_metadata(split)
        # Sample 20 rows for speed
        if len(df) > 20:
            df = df.sample(n=20, random_state=Config.SEED).reset_index(drop=True)
        return df

    # Apply mocks to modules that import and use get_metadata
    feature_ext.get_metadata = mock_get_metadata
    ranker_lib.get_metadata = mock_get_metadata
    inference_lib.get_metadata = mock_get_metadata
    pair_gen.get_metadata = mock_get_metadata

    print("Configuration and Mocking complete.")

    # 3. Test Data Utils
    print("\n--- Testing Data Utils ---")
    df_train = mock_get_metadata("train")
    assert len(df_train) <= 20
    sample_id = df_train.iloc[0]["id"]
    sample_path = df_train.iloc[0]["file_path"]

    cells = data_utils.get_notebook_cells(sample_id, sample_path)
    assert "code_cells" in cells
    assert "markdown_cells" in cells
    assert isinstance(cells["code_cells"], list)
    print("Data Utils: Notebook loading verified.")

    # 4. Test Pair Generation
    print("\n--- Testing Pair Generation ---")
    # Force reload to ignore any existing cache
    df_pairs = pair_gen.generate_bidirectional_pairs(load_cached_data=False)

    assert not df_pairs.empty, "Pair generation produced empty DataFrame"
    assert "markdown" in df_pairs.columns
    assert "code" in df_pairs.columns
    print(f"Pair Generation: Generated {len(df_pairs)} pairs.")

    # 5. Test Fine-Tuning
    print("\n--- Testing Fine-Tuning ---")
    ft = fine_tuning.FineTuner()
    # Ensure it uses the patched config values
    assert ft.epochs == 1

    ft.run()

    # Verify output
    assert os.path.exists(Config.Paths.MODEL_OUTPUT_DIR)
    # Check for model file (SentenceTransformers usually saves config.json and model.safetensors/bin)
    files = os.listdir(Config.Paths.MODEL_OUTPUT_DIR)
    has_model = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
    assert has_model, "Fine-tuned model file not found."
    print("Fine-Tuning: Model saved successfully.")

    # 6. Test Feature Extraction
    print("\n--- Testing Feature Extraction ---")
    extractor = feature_ext.FeatureExtractor()
    # We use the mock_get_metadata, so this processes only ~20 notebooks
    df_feats = extractor.extract_features("train", load_cached_data=False)

    assert not df_feats.empty
    assert "heatmap_0" in df_feats.columns
    assert "target" in df_feats.columns
    print(f"Feature Extraction: Generated features with shape {df_feats.shape}.")

    # 7. Test Ranker (Training & Submission)
    print("\n--- Testing LGBM Ranker ---")
    ranker = ranker_lib.LGBMRanker()

    # Train (this will also trigger feature extraction for val set)
    ranker.train()
    assert os.path.exists(ranker.model_path), "LGBM model file not found."

    # Generate Submission (uses test split)
    ranker.generate_submission()
    assert os.path.exists(ranker.submission_path), "Submission file not found."

    df_sub = pd.read_csv(ranker.submission_path)
    assert "id" in df_sub.columns
    assert "cell_order" in df_sub.columns
    assert len(df_sub) > 0
    print("LGBM Ranker: Training and Submission generation successful.")

    # 8. Test Inference (Single Notebook)
    print("\n--- Testing Single Inference ---")
    # Pick a test ID from the mocked metadata
    df_test = mock_get_metadata("test")
    test_id = df_test.iloc[0]["id"]

    # Run prediction
    predicted_order = inference_lib.predict_order(test_id, ranker, extractor)

    assert isinstance(predicted_order, str)
    # The order should contain at least one cell ID (unless notebook is empty, which is rare)
    if len(predicted_order) > 0:
        assert len(predicted_order.split()) > 0

    print(f"Inference: Successfully predicted order for {test_id}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
