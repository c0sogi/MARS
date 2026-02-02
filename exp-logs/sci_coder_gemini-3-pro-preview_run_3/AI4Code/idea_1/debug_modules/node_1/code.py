import os
import shutil
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import load_metadata, load_notebook
from library.utils import preprocess_text, compute_kendall_tau
from library.features import SemanticFeatureExtractor
from library.model import PositionRegressor


def set_seed(seed=42):
    np.random.seed(seed)
    # If torch were used, we would set it here too.


def main():
    print("=== Notebook Cell Ordering Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast execution...")

    # Override Config defaults to use a tiny subset and fast training parameters
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks
    Config.MODEL_PARAMS["n_estimators"] = 10  # Very few trees for demo
    Config.MODEL_PARAMS["verbose"] = -1
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = -1  # Silent evaluation

    # Ensure working directories are clean/ready
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    Config.setup()

    set_seed(Config.SEED)
    print("   Configuration complete.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n2. Demonstrating Data Loading...")

    # Load metadata subsets
    df_train = load_metadata("train", sample_size=Config.DEBUG_SAMPLE_SIZE)
    df_val = load_metadata("val", sample_size=Config.DEBUG_SAMPLE_SIZE)
    df_test = load_metadata("test", sample_size=Config.DEBUG_SAMPLE_SIZE)

    print(f"   Loaded Train Metadata: {df_train.shape}")
    print(f"   Loaded Val Metadata:   {df_val.shape}")
    print(f"   Loaded Test Metadata:  {df_test.shape}")

    assert not df_train.empty, "Train metadata is empty."
    assert not df_val.empty, "Validation metadata is empty."

    # Demonstrate loading a single notebook content
    sample_id = df_train.iloc[0]["id"]
    sample_path = df_train.iloc[0]["file_path"]
    nb_content = load_notebook(sample_path)

    print(f"   Sample Notebook ({sample_id}):")
    print(f"     - Code Cells: {len(nb_content['code_cells'])}")
    print(f"     - Markdown Cells: {len(nb_content['markdown_cells'])}")

    assert isinstance(nb_content, dict), "load_notebook should return a dictionary."
    assert "code_cells" in nb_content and "markdown_cells" in nb_content

    # -------------------------------------------------------------------------
    # 3. Utility Function Demonstration
    # -------------------------------------------------------------------------
    print("\n3. Demonstrating Text Preprocessing...")

    raw_text = "Imports: import numpy as np; # This is a COMMENT!"
    clean_text = preprocess_text(raw_text, stem=True)
    print(f"   Raw:   '{raw_text}'")
    print(f"   Clean: '{clean_text}'")

    # Basic check: 'numpy' should be kept, 'COMMENT' should be lowercased/stemmed
    assert "numpi" in clean_text
    assert "comment" in clean_text or "com" in clean_text

    # -------------------------------------------------------------------------
    # 4. Feature Extraction Demonstration
    # -------------------------------------------------------------------------
    print("\n4. Demonstrating Feature Extraction...")

    extractor = SemanticFeatureExtractor()

    # Fit TF-IDF Vectorizer on the small training subset
    # load_cached=False forces a fresh fit
    extractor.fit_vectorizer(
        df_train, sample_size=Config.DEBUG_SAMPLE_SIZE, load_cached=False
    )
    assert os.path.exists(extractor.vectorizer_path), "Vectorizer file was not saved."
    print("   Vectorizer fitted and saved.")

    # Generate features for Train and Val
    # This also saves the parquet files to cache, which the model trainer will pick up
    df_train_feats = extractor.generate_dataset(
        df_train, mode="train", load_cached_data=False
    )
    df_val_feats = extractor.generate_dataset(
        df_val, mode="val", load_cached_data=False
    )

    print(f"   Train Features Shape: {df_train_feats.shape}")

    # Check feature structure
    if not df_train_feats.empty:
        expected_cols = ["id", "cell_id", "n_code", "sim_mean", "target"]
        for col in expected_cols:
            assert col in df_train_feats.columns, f"Missing expected column: {col}"
    else:
        print(
            "   (Warning: Sampled notebooks had no markdown cells to extract features from.)"
        )

    # -------------------------------------------------------------------------
    # 5. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Demonstrating Model Training...")

    regressor = PositionRegressor()

    # Train the model using the metadata
    # Note: The regressor will look for cached feature files generated in step 4
    regressor.train(df_train, df_val)

    assert os.path.exists(regressor.model_path), "Model file was not saved."
    assert (
        regressor.model is not None
    ), "Regressor model attribute is None after training."
    print("   Model training complete and saved.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n6. Demonstrating Inference and Submission...")

    # Predict on test set
    preds = regressor.predict(df_test)
    print(f"   Predictions generated. Shape: {preds.shape}")

    # Generate submission file
    regressor.generate_submission(df_test)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission loaded. Shape: {df_sub.shape}")
    print(f"   First 2 rows:\n{df_sub.head(2)}")

    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect."
    assert len(df_sub) == len(df_test), "Submission row count mismatch."

    # -------------------------------------------------------------------------
    # 7. Metric Calculation Demonstration
    # -------------------------------------------------------------------------
    print("\n7. Demonstrating Metric Calculation (Kendall Tau)...")

    # We will use the training data as "Ground Truth" and create dummy predictions
    df_truth = df_train[["id", "cell_order"]].copy()

    # Case A: Perfect Prediction
    df_perfect = df_truth.copy()
    score_perfect = compute_kendall_tau(df_perfect, df_truth)
    print(f"   Perfect Score (Expected ~1.0): {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "Perfect prediction did not score 1.0"

    # Case B: Reversed Prediction (Simulating bad performance)
    df_reversed = df_truth.copy()
    df_reversed["cell_order"] = df_reversed["cell_order"].apply(
        lambda x: " ".join(x.split()[::-1])
    )
    score_reversed = compute_kendall_tau(df_reversed, df_truth)
    print(f"   Reversed Score (Expected < 1.0): {score_reversed:.4f}")
    assert score_reversed < 1.0, "Reversed prediction should score less than 1.0"

    print("\n=== All demonstrations passed successfully! ===")


if __name__ == "__main__":
    main()
