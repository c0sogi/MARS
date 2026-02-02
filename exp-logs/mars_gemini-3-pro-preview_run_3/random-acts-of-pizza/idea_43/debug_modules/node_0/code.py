import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.data_loader import load_datasets, get_target, get_metadata
from library.feature_generators import (
    TextProcessor,
    LatentCommunityInjector,
    MetadataAugmenter,
    SentenceEmbedder,
)
from library.model_definitions import get_base_learner, get_meta_learner
from library.stacking_manager import HexStackEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def test_data_loader():
    print("\n=== Testing Data Loader ===")
    # Load debug subset
    train_df, val_df, test_df = load_datasets(debug=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Assertions
    assert not train_df.empty, "Train dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"

    return train_df


def test_feature_generators(df):
    print("\n=== Testing Feature Generators ===")

    # 1. Text Processor
    print("Testing TextProcessor...")
    tp = TextProcessor()
    text_series = tp.process(df)
    assert len(text_series) == len(df), "TextProcessor output length mismatch"
    assert isinstance(text_series.iloc[0], str), "TextProcessor output is not string"
    print("TextProcessor passed.")

    # 2. Latent Community Injector (NMF)
    print("Testing LatentCommunityInjector...")
    injector = LatentCommunityInjector()
    # Mock fit on small data
    injector.fit(df)
    nmf_features = injector.transform(df)
    assert nmf_features.shape == (
        len(df),
        Config.NMF_COMPONENTS,
    ), f"NMF output shape mismatch. Expected {(len(df), Config.NMF_COMPONENTS)}, got {nmf_features.shape}"
    print("LatentCommunityInjector passed.")

    # 3. Metadata Augmenter
    print("Testing MetadataAugmenter...")
    augmenter = MetadataAugmenter()
    augmenter.fit(df)
    # Transform with NMF features
    meta_features = augmenter.transform(df, nmf_features=nmf_features)
    # Check shape: numerical metadata cols + NMF components
    # We don't hardcode exact metadata count here as it depends on get_metadata logic,
    # but we verify it's > NMF components
    assert (
        meta_features.shape[1] > Config.NMF_COMPONENTS
    ), "Metadata augmentation failed to add features"
    assert meta_features.shape[0] == len(df), "Metadata output row count mismatch"
    print("MetadataAugmenter passed.")

    # 4. Sentence Embedder
    print("Testing SentenceEmbedder...")
    embedder = SentenceEmbedder()
    # Use a very small slice for speed
    small_text = text_series.head(5)
    embeddings = embedder.transform(small_text, load_cached_data=False)
    assert embeddings.shape == (
        5,
        384,
    ), f"Embedding shape mismatch. Expected (5, 384), got {embeddings.shape}"
    print("SentenceEmbedder passed.")


def test_model_definitions():
    print("\n=== Testing Model Definitions ===")

    # Test instantiation of a base learner
    rf = get_base_learner("lexical_bagger", n_estimators=10)
    assert hasattr(rf, "fit") and hasattr(
        rf, "predict_proba"
    ), "Base learner missing sklearn interface"

    # Test instantiation of meta learner
    meta = get_meta_learner()
    assert hasattr(meta, "fit"), "Meta learner missing fit method"
    print("Model definitions passed.")


def test_stacking_pipeline():
    print("\n=== Testing HexStackEnsemble Pipeline ===")

    # Initialize Manager
    manager = HexStackEnsemble()

    # 1. Train OOF (Level 1)
    print("Running OOF Training (Debug Mode)...")
    # This runs 5-fold CV on the debug dataset
    oof_preds, y_true = manager.train_oof(debug=True)

    assert len(oof_preds) == len(y_true), "OOF predictions length mismatch"
    assert (
        oof_preds.shape[1] == 6
    ), "OOF predictions should have 6 columns (one per base learner)"

    # Check if AUC can be calculated (just to ensure values are valid probabilities)
    try:
        score = roc_auc_score(y_true, oof_preds.mean(axis=1))
        print(f"OOF AUC (Debug): {score:.4f}")
    except ValueError:
        print("Skipping AUC calculation (likely single class in debug fold).")

    # 2. Train Meta Learner (Level 2)
    print("Training Meta Learner...")
    manager.train_meta(oof_preds, y_true)
    assert manager.meta_learner is not None, "Meta learner not stored after training"

    # 3. Final Retraining
    print("Running Final Retraining (Debug Mode)...")
    manager.retrain_final(debug=True)
    assert manager.final_transformers_fitted, "Final transformers flag not set"
    assert (
        "lexical_bagger" in manager.base_learners
    ), "Base learners not stored after retraining"

    # 4. Prediction
    print("Running Prediction on Test Set (Debug Mode)...")
    manager.predict(debug=True)

    # Verify Submission File
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")
    assert "request_id" in sub_df.columns, "request_id column missing in submission"
    assert (
        "requester_received_pizza" in sub_df.columns
    ), "prediction column missing in submission"
    assert not sub_df.empty, "Submission file is empty"

    print("Pipeline execution successful.")


if __name__ == "__main__":
    set_seed(42)

    # Ensure working directories exist
    Config.ensure_directories()

    try:
        # 1. Data Loading
        df_train = test_data_loader()

        # 2. Unit Tests for Feature Generators
        test_feature_generators(df_train)

        # 3. Unit Tests for Model Factory
        test_model_definitions()

        # 4. Integration Test for Full Pipeline
        test_stacking_pipeline()

        print("\nALL TESTS PASSED SUCCESSFULLY.")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
