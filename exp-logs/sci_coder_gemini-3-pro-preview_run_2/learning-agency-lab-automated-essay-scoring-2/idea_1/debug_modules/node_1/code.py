import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, save_artifact, load_artifact
from library.metrics import compute_qwk
from library.data_loader import load_data
from library.features import TextVectorizer
from library.model import ScoreRegressor
from library.train import run_training
from library.inference import generate_submission


def run_demo():
    print("=== Essay Scoring Pipeline Demonstration ===")

    # 1. Setup & Utils Verification
    print("\n[1/7] Setting up and verifying utilities...")
    seed_everything(42)
    Config.create_dirs()

    # Test artifact saving/loading
    test_obj = {"hello": "world", "values": [1, 2, 3]}
    test_path = os.path.join(Config.WORKING_DIR, "test_artifact.pkl")
    save_artifact(test_obj, test_path)
    loaded_obj = load_artifact(test_path)

    assert loaded_obj == test_obj, "Artifact save/load failed"
    print("Utils verification passed.")

    # 2. Metrics Verification
    print("\n[2/7] Verifying metrics (Quadratic Weighted Kappa)...")
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred_perfect = np.array([1, 2, 3, 4, 5, 6])
    y_pred_bad = np.array([6, 5, 4, 3, 2, 1])

    qwk_perfect = compute_qwk(y_true, y_pred_perfect)
    qwk_bad = compute_qwk(y_true, y_pred_bad)

    print(f"QWK (Perfect Match): {qwk_perfect}")
    print(f"QWK (Inverse Match): {qwk_bad}")

    assert np.isclose(qwk_perfect, 1.0), "QWK calculation incorrect for perfect match"
    assert qwk_bad < 0.5, "QWK calculation incorrect for bad match"
    print("Metrics verification passed.")

    # 3. Data Loader Verification
    print("\n[3/7] Verifying Data Loader...")
    # Load a small subset to verify schema and loading logic
    N_ROWS = 100
    df_train = load_data("train", nrows=N_ROWS, load_cached_data=False)

    assert len(df_train) == N_ROWS, f"Expected {N_ROWS} rows, got {len(df_train)}"
    assert Config.TEXT_COL in df_train.columns, f"Missing {Config.TEXT_COL} column"
    assert Config.TARGET_COL in df_train.columns, f"Missing {Config.TARGET_COL} column"

    # Verify preprocessing (checking if text is string)
    assert isinstance(
        df_train.iloc[0][Config.TEXT_COL], str
    ), "Text column content is not string"
    print(f"Loaded {len(df_train)} rows. Columns: {list(df_train.columns)}")
    print("Data Loader verification passed.")

    # 4. Feature Extraction Component Verification
    print("\n[4/7] Verifying Feature Extraction (TextVectorizer)...")
    texts = df_train[Config.TEXT_COL].tolist()

    # Initialize vectorizer with small limits for quick testing
    # Note: In the real pipeline, parameters come from Config
    vec = TextVectorizer(max_features=50, ngram_range=(1, 1))
    X_train_small = vec.fit_transform(texts)

    print(f"Extracted features shape: {X_train_small.shape}")
    assert X_train_small.shape[0] == N_ROWS
    assert X_train_small.shape[1] <= 50

    # Test save/load of vectorizer
    vec_path = os.path.join(Config.WORKING_DIR, "test_vec.joblib")
    vec.save(vec_path)
    vec_loaded = TextVectorizer.load(vec_path)
    X_train_loaded = vec_loaded.transform(texts)

    # Check if transformation is identical (sparse matrix comparison)
    # Using tolerance because fit_transform vs transform can have minor float diffs
    diff = np.abs(X_train_small - X_train_loaded).max()
    assert (
        diff < 1e-7
    ), f"Loaded vectorizer produced different results (max diff: {diff})"
    print("Feature Extraction component verification passed.")

    # 5. Model Component Verification
    print("\n[5/7] Verifying Model (ScoreRegressor)...")
    y_train_small = df_train[Config.TARGET_COL].values

    # Initialize regressor with specific alpha to test basic fit/predict
    model = ScoreRegressor(alpha=1.0)

    # Train on the small feature set
    model.train(X_train_small, y_train_small)

    # Predict
    preds = model.predict(X_train_small)

    print(f"Predictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")

    assert len(preds) == N_ROWS
    assert (
        preds.min() >= Config.SCORE_MIN
    ), f"Prediction below min score {Config.SCORE_MIN}"
    assert (
        preds.max() <= Config.SCORE_MAX
    ), f"Prediction above max score {Config.SCORE_MAX}"
    print("Model component verification passed.")

    # 6. Full Training Pipeline Integration
    print("\n[6/7] Running Full Training Pipeline (Integration Test)...")
    # Using a subset (nrows=500) to ensure the pipeline runs quickly but exercises all logic
    PIPELINE_ROWS = 500

    # run_training handles loading data, feature extraction, model training (with validation), and saving artifacts
    run_training(load_cached_data=False, nrows=PIPELINE_ROWS)

    # Verify artifacts were created
    assert os.path.exists(
        Config.VECTORIZER_PATH
    ), f"Vectorizer not found at {Config.VECTORIZER_PATH}"
    assert os.path.exists(Config.MODEL_PATH), f"Model not found at {Config.MODEL_PATH}"
    print("Training pipeline executed successfully.")

    # 7. Inference Pipeline Integration
    print("\n[7/7] Running Inference Pipeline (Integration Test)...")
    # generate_submission handles loading test data, loading artifacts, predicting, and saving CSV
    generate_submission(load_cached_data=False, nrows=PIPELINE_ROWS)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    assert Config.ID_COL in df_sub.columns
    assert Config.TARGET_COL in df_sub.columns
    # Ensure scores are integers as required by the metric/submission format
    assert pd.api.types.is_integer_dtype(
        df_sub[Config.TARGET_COL]
    ), "Submission scores are not integers"

    # Check value range
    valid_scores = set(range(Config.SCORE_MIN, Config.SCORE_MAX + 1))
    assert (
        df_sub[Config.TARGET_COL].isin(valid_scores).all()
    ), "Invalid scores found in submission"

    print("Inference pipeline executed successfully.")
    print("\n=== All Demonstrations Passed ===")


if __name__ == "__main__":
    run_demo()
