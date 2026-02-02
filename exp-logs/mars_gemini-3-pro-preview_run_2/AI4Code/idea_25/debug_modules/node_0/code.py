import os
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import library components
from library.config import WORKING_DIR, SUBMISSION_DIR
from library.utils import seed_everything, kendall_tau_metric
from library.data_loader import load_data
from library.text_processing import get_vectorizer, TextVectorizer
from library.models import Stage1Ridge, Stage2LGBM
from library.feature_extraction import NeighborhoodExtractor, assemble_stage2_features
from library.pipeline import train_pipeline, inference_pipeline


def clean_working_dir():
    """Helper to clean up working directory for a fresh demo run."""
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)


def demo_utils():
    print("\n--- Demo: Utils (Kendall Tau Metric) ---")
    # Scenario: 1 Notebook, 4 cells.
    # Ground Truth: A B C D
    # Prediction: A C B D (1 swap needed: C and B)
    # Total pairs: n(n-1) = 4*3 = 12
    # Inversions (swaps): 1
    # Score = 1 - 4 * (1 / 12) = 1 - 1/3 = 0.6667

    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["A B C D"]})
    df_pred = pd.DataFrame({"id": ["nb1"], "cell_order": ["A C B D"]})

    score = kendall_tau_metric(df_pred, df_gt)
    print(f"Calculated Score: {score:.4f}")

    # Assert correctness
    expected_score = 1.0 - 4.0 * (1.0 / 12.0)
    assert np.isclose(
        score, expected_score
    ), f"Metric mismatch. Expected {expected_score}, got {score}"
    print("Utils verification passed.")


def demo_data_loading():
    print("\n--- Demo: Data Loading ---")
    # Load a tiny subset (debug_n=50 notebooks)
    # We force load_cached_data=False to demonstrate processing logic,
    # though in production True is preferred.
    df_train = load_data("train", load_cached_data=False, debug_n=50)

    print(f"Loaded train data shape: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)}")

    # Basic assertions
    assert not df_train.empty, "Train dataframe is empty."
    assert "pct_rank" in df_train.columns, "Target column 'pct_rank' missing in train."
    assert "source" in df_train.columns, "Source text column missing."

    return df_train


def demo_vectorization(df_train):
    print("\n--- Demo: Text Vectorization ---")
    # Instantiate vectorizer
    # We use a factory method that handles caching logic
    vectorizer = get_vectorizer(train_texts=df_train["source"], load_cached=False)

    # Transform data
    tfidf_mat, svd_mat = vectorizer.transform(df_train["source"])

    print(f"TF-IDF Matrix shape: {tfidf_mat.shape}")
    print(f"SVD Matrix shape: {svd_mat.shape}")

    # Assertions
    assert tfidf_mat.shape[0] == len(df_train), "TF-IDF row count mismatch."
    assert svd_mat.shape[0] == len(df_train), "SVD row count mismatch."
    assert (
        svd_mat.shape[1] == vectorizer.svd_n_components
    ), "SVD component count mismatch."

    return vectorizer, tfidf_mat


def demo_stage1_ridge(df_train, tfidf_mat):
    print("\n--- Demo: Stage 1 (Ridge Regression) ---")
    model = Stage1Ridge(alpha=1.0)

    y = df_train["pct_rank"].values
    cell_ids = df_train["cell_id"].values

    # Fit model
    model.fit(tfidf_mat, y)

    # Generate OOF predictions (2 folds for speed)
    df_oof = model.get_oof_predictions(
        tfidf_mat, y, cell_ids, n_splits=2, load_cached_data=False
    )

    print("OOF Predictions head:")
    print(df_oof.head())

    # Assertions
    assert len(df_oof) == len(df_train), "OOF prediction count mismatch."
    assert "ridge_rank" in df_oof.columns, "Prediction column missing."

    # Check correlation (should be positive even for weak model)
    corr = np.corrcoef(df_oof["ridge_rank"], y)[0, 1]
    print(f"Stage 1 Correlation with Target: {corr:.4f}")

    return df_oof


def demo_feature_extraction(df_train, vectorizer):
    print("\n--- Demo: Feature Extraction (Neighborhoods) ---")
    extractor = NeighborhoodExtractor(num_neighbors=5, top_k=2)

    # Extract features
    # This computes similarity between Markdown and Code cells
    df_neigh = extractor.extract_neighborhood_features(
        df_train, vectorizer, split="train", load_cached_data=False
    )

    print(f"Neighborhood Features shape: {df_neigh.shape}")
    print(f"Feature columns: {list(df_neigh.columns[:5])} ...")

    # Assertions
    # Only markdown cells should be in the output
    n_md_input = df_train[df_train["cell_type"] == "markdown"].shape[0]
    assert (
        len(df_neigh) == n_md_input
    ), f"Feature count {len(df_neigh)} matches MD count {n_md_input}"
    assert "lex_top1_sim" in df_neigh.columns, "Lexical feature missing."
    assert "lat_top1_sim" in df_neigh.columns, "Latent feature missing."

    return df_neigh


def demo_stage2_lgbm(df_train, df_neigh, df_oof):
    print("\n--- Demo: Stage 2 (LightGBM) ---")

    # 1. Assemble features
    df_stage2 = assemble_stage2_features(df_neigh, df_oof, df_train)

    # Merge target
    df_stage2 = pd.merge(
        df_stage2, df_train[["cell_id", "pct_rank"]], on="cell_id", how="inner"
    )

    print(f"Stage 2 Dataset shape: {df_stage2.shape}")

    # 2. Prepare X, y
    drop_cols = ["cell_id", "notebook_id", "pct_rank"]
    features = [c for c in df_stage2.columns if c not in drop_cols]

    X = df_stage2[features]
    y = df_stage2["pct_rank"]

    # 3. Fit Model
    lgbm_wrapper = Stage2LGBM()
    # Reduce estimators for demo speed
    lgbm_wrapper.model.set_params(n_estimators=10)

    lgbm_wrapper.fit(X, y)

    # 4. Predict
    preds = lgbm_wrapper.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"Stage 2 Training MAE: {mae:.4f}")

    assert len(preds) == len(y), "Prediction length mismatch."


def demo_full_pipeline():
    print("\n--- Demo: Full Pipeline Execution ---")
    # This runs the high-level functions provided in pipeline.py
    # We use a small debug_n to keep it fast.

    print(">> Running Train Pipeline...")
    val_score = train_pipeline(debug_n=50, load_cached_data=False)
    print(f"Pipeline Validation Score: {val_score:.4f}")

    print(">> Running Inference Pipeline...")
    # Inference pipeline loads the models saved by train_pipeline
    # and generates submission.csv
    inference_pipeline(debug_n=20, load_cached_data=False)

    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())
    assert df_sub.shape[1] == 2, "Submission should have 2 columns (id, cell_order)"


if __name__ == "__main__":
    # Set global seed
    seed_everything(42)

    # Clean up before starting
    clean_working_dir()

    try:
        # 1. Verify Metric
        demo_utils()

        # 2. Load Data
        df_train = demo_data_loading()

        # 3. Vectorize
        vectorizer, tfidf_mat = demo_vectorization(df_train)

        # 4. Stage 1 Ridge
        df_oof = demo_stage1_ridge(df_train, tfidf_mat)

        # 5. Feature Extraction
        df_neigh = demo_feature_extraction(df_train, vectorizer)

        # 6. Stage 2 LGBM
        demo_stage2_lgbm(df_train, df_neigh, df_oof)

        # 7. End-to-End Pipeline
        # Note: This will re-run steps but ensures the integrated logic in pipeline.py is correct
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        raise e
