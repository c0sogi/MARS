import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, decode_text
from library.data_loader import load_datasets
from library.nb_transformer import NBTransformer
from library.model import NBLRModel


def main():
    print("Initializing demonstration...")

    # ==========================================
    # 1. Configuration Override (Optimize for Speed)
    # ==========================================
    # We monkey-patch the Config class to use a small subset of data
    # and limit iterations for rapid execution.
    print("Configuring for fast execution...")
    Config.DEBUG = True
    Config.MAX_TRAIN_SAMPLES = 200  # Use only 200 samples
    Config.LR_MAX_ITER = 20  # Limit logistic regression iterations
    Config.WORD_MAX_FEATURES = 500  # Limit vocabulary size
    Config.CHAR_MAX_FEATURES = 500

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("Verifying utilities...")
    # Test decode_text
    raw_text = "Hello\\u0020World"
    decoded = decode_text(raw_text)
    assert (
        decoded == "Hello World"
    ), f"decode_text failed: expected 'Hello World', got '{decoded}'"

    # Test NaN handling
    assert decode_text(np.nan) == "", "decode_text failed on NaN"
    print("Utilities verified.")

    # ==========================================
    # 3. Verify Data Loader
    # ==========================================
    print("Loading datasets (subset)...")
    # We force load_cached_data=False to ensure we test the raw loading logic
    # and apply our MAX_TRAIN_SAMPLES limit.
    train_df, val_df, test_df = load_datasets(debug=True, load_cached_data=False)

    # Assertions
    assert train_df is not None, "Train DF is None"
    assert val_df is not None, "Val DF is None"
    assert test_df is not None, "Test DF is None"

    # Check dimensions (should be limited by MAX_TRAIN_SAMPLES)
    assert (
        len(train_df) <= Config.MAX_TRAIN_SAMPLES
    ), f"Train size {len(train_df)} exceeds limit"
    assert (
        len(val_df) <= Config.MAX_TRAIN_SAMPLES
    ), f"Val size {len(val_df)} exceeds limit"

    # Check columns
    required_cols = [Config.TEXT_COL, Config.DATE_COL]
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in train"
        assert col in test_df.columns, f"Missing column {col} in test"

    print(f"Data loaded successfully. Train shape: {train_df.shape}")

    # ==========================================
    # 4. Verify NBTransformer Logic
    # ==========================================
    print("Verifying NBTransformer...")
    # Create synthetic binary classification data
    # 4 samples, 3 features
    X_dummy = sparse.csr_matrix(
        [
            [1, 0, 1],  # Class 1
            [1, 1, 0],  # Class 1
            [0, 1, 0],  # Class 0
            [0, 0, 1],  # Class 0
        ],
        dtype=np.float64,
    )
    y_dummy = np.array([1, 1, 0, 0])

    nb = NBTransformer(alpha=1.0)
    nb.fit(X_dummy, y_dummy)

    # Transform
    X_trans = nb.transform(X_dummy)

    # Checks
    assert sparse.issparse(X_trans), "Transformed data should be sparse"
    assert X_trans.shape == X_dummy.shape, "Shape mismatch after transformation"
    # Check that weights are not all 1.0 (transformation happened)
    # Since X is binary, X_trans will have values equal to the r vector where X was 1.
    # We just check it's not identical to input (unless r happens to be exactly 1s, which is unlikely here)
    assert not np.allclose(
        X_trans.toarray(), X_dummy.toarray()
    ), "Transformation had no effect"
    print("NBTransformer verified.")

    # ==========================================
    # 5. Verify NBLRModel (Training & Inference)
    # ==========================================
    print("Training NBLRModel on subset...")
    model = NBLRModel()

    # Fit model
    auc_score = model.fit(train_df, val_df)

    # Verify Metric
    print(f"Model trained. Validation AUC: {auc_score:.4f}")
    assert isinstance(auc_score, float), "AUC score is not a float"
    assert 0.0 <= auc_score <= 1.0, f"AUC score {auc_score} out of range"

    # Prediction
    print("Generating predictions...")
    preds = model.predict_proba(test_df)

    # Verify Predictions
    assert len(preds) == len(test_df), "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0,1]"

    # ==========================================
    # 6. Verify Model Persistence
    # ==========================================
    print("Saving model artifact...")
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.joblib")
    model.save(save_path)
    assert os.path.exists(save_path), "Model file was not saved"
    print("Model saved successfully.")

    # ==========================================
    # 7. Generate Submission File
    # ==========================================
    print("Generating submission file...")
    submission_df = test_df.copy()
    submission_df["Insult"] = preds

    # Format according to requirements
    cols = ["Insult", Config.DATE_COL, Config.TEXT_COL]
    submission_df = submission_df[cols]

    sub_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path), "Submission file not created"

    # Verify content format
    saved_df = pd.read_csv(sub_path)
    assert list(saved_df.columns) == cols, "Submission columns mismatch"
    assert saved_df.shape[0] == len(test_df), "Submission row count mismatch"

    print("Submission generated successfully.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
