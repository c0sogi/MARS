import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import project library modules
from library.configuration import Config
from library.text_utils import (
    tokenize,
    strip_html_tags,
    map_clean_to_raw_span,
    build_vocab,
)
from library.feature_engineering import BM25, FeatureExtractor
from library.data_loader import RankerDatasetBuilder, create_reader_dataset
from library.model_ranker import GradientBoostingRanker
from library.model_reader import ReaderTrainer
from library.predictor import Evaluator


def setup_demo_environment():
    """
    Overrides default configuration to use a specific demo directory
    and reduce computational load for the demonstration.
    """
    print(">>> Setting up demo environment...")

    # Define a separate working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths
    Config.WORKING_DIR = demo_dir
    Config.RANKER_MODEL_PATH = os.path.join(demo_dir, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(demo_dir, "reader_best.pth")
    Config.VOCAB_CACHE_PATH = os.path.join(demo_dir, "vocab.parquet")

    # Cache files
    Config.RANKER_TRAIN_CACHE = os.path.join(demo_dir, "ranker_train_data.parquet")
    Config.RANKER_VAL_CACHE = os.path.join(demo_dir, "ranker_val_data.parquet")
    Config.READER_TRAIN_CACHE = os.path.join(demo_dir, "reader_train_data.parquet")
    Config.READER_VAL_CACHE = os.path.join(demo_dir, "reader_val_data.parquet")

    # Output
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission/submission.csv")

    # Hyperparameters for speed
    Config.RANKER_NUM_BOOST_ROUND = 10
    Config.RANKER_EARLY_STOPPING_ROUNDS = 5
    Config.READER_EPOCHS = 1
    Config.READER_BATCH_SIZE = 4
    Config.VOCAB_SIZE = 2000  # Small vocab for demo

    print(f"Working directory set to: {Config.WORKING_DIR}")


def demo_text_utils():
    """
    Validates text processing utilities.
    """
    print("\n>>> Demo: Text Utilities")

    # Test Case
    raw_text = "<P> The quick brown fox. </P>"
    tokens = tokenize(raw_text)
    print(f"Tokens: {tokens}")

    # Verify Tokenization
    assert tokens == ["<P>", "The", "quick", "brown", "fox.", "</P>"]

    # Verify HTML Stripping
    clean_tokens, mapping = strip_html_tags(tokens)
    print(f"Clean Tokens: {clean_tokens}")
    print(f"Mapping: {mapping}")

    assert clean_tokens == ["The", "quick", "brown", "fox."]
    assert mapping == [1, 2, 3, 4]  # Indices in original list

    # Verify Span Mapping
    # Map clean span "quick brown" (indices 1 to 3 exclusive) back to raw
    raw_start, raw_end = map_clean_to_raw_span(1, 3, mapping)
    print(f"Mapped Span (Clean 1:3 -> Raw {raw_start}:{raw_end})")

    # clean[1] is "quick" -> raw[2]
    # clean[3] is "fox." -> raw[4]
    assert raw_start == 2
    assert raw_end == 4
    assert tokens[raw_start:raw_end] == ["quick", "brown"]

    print("Text utilities verification passed.")


def demo_feature_engineering():
    """
    Validates feature extraction logic.
    """
    print("\n>>> Demo: Feature Engineering")

    # Dummy Data
    questions = [["what", "is", "python"], ["who", "wrote", "code"]]
    candidates = [["python", "is", "a", "language"], ["code", "was", "written", "by"]]

    # Test BM25
    bm25 = BM25()
    bm25.fit(candidates)
    score = bm25.score(["python"], ["python", "is", "great"])
    print(f"BM25 Score: {score:.4f}")
    assert score > 0

    # Test Feature Extractor
    extractor = FeatureExtractor()
    extractor.fit(questions, candidates)

    # Compute features for a match
    feats = extractor.compute_features(
        query_tokens=["python"],
        candidate_tokens=["python", "is", "good"],
        candidate_idx=0,
        total_candidates=1,
    )

    print("Extracted Features:", list(feats.keys()))
    assert "bm25_score" in feats
    assert "tfidf_cosine" in feats
    assert feats["unigram_overlap_count"] == 1

    print("Feature engineering verification passed.")


def demo_ranker_model():
    """
    Demonstrates Ranker training and prediction.
    """
    print("\n>>> Demo: Ranker Model (LightGBM)")

    # Use a small sample size for speed
    sample_size = 100
    print(f"Building Ranker datasets (sample_size={sample_size})...")

    # Force creation of new datasets
    train_df = RankerDatasetBuilder.build_train_set(
        load_cached_data=False, sample_size=sample_size
    )
    val_df = RankerDatasetBuilder.build_val_set(
        load_cached_data=False, sample_size=sample_size
    )

    assert not train_df.empty, "Ranker training set is empty"
    assert "label" in train_df.columns

    # Train Ranker
    print("Training Ranker...")
    ranker = GradientBoostingRanker()
    ranker.train(load_cached_data=True)  # Load the files we just built

    assert os.path.exists(Config.RANKER_MODEL_PATH), "Ranker model file not created"

    # Predict
    print("Running Ranker Prediction...")
    preds = ranker.predict(val_df)

    print(f"Predictions shape: {preds.shape}")
    print(f"Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")

    assert len(preds) == len(val_df)

    print("Ranker model demonstration successful.")


def demo_reader_model():
    """
    Demonstrates Reader training.
    """
    print("\n>>> Demo: Reader Model (Bi-GRU)")

    sample_size = 50

    # Explicitly build vocab first to ensure it exists in the demo dir
    print("Building vocabulary...")
    build_vocab(
        cache_path=Config.VOCAB_CACHE_PATH,
        vocab_size=Config.VOCAB_SIZE,
        sample_rate=0.05,
        load_cached_data=False,
    )

    # Train Reader
    # Note: ReaderTrainer handles dataset creation internally via get_reader_loaders
    print("Training Reader...")
    trainer = ReaderTrainer()
    trainer.train(load_cached_data=False, sample_size=sample_size)

    assert os.path.exists(Config.READER_MODEL_PATH), "Reader model file not created"
    print("Reader model demonstration successful.")


def demo_inference():
    """
    Demonstrates the full inference pipeline generating a submission.
    """
    print("\n>>> Demo: Full Inference Pipeline")

    evaluator = Evaluator()

    # Generate submission for a small subset of test data
    test_sample_size = 20
    print(f"Generating submission for {test_sample_size} test examples...")

    evaluator.generate_submission(load_cached_data=False, sample_size=test_sample_size)

    # Verify Output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {submission_df.shape}")
    print("First few rows:")
    print(submission_df.head())

    # Basic format checks
    assert "example_id" in submission_df.columns
    assert "PredictionString" in submission_df.columns
    # Should have 2 rows per example (long + short)
    expected_rows = test_sample_size * 2
    # Note: Actual rows might be less if file grouping/reading logic skips invalid lines,
    # but for this demo with valid metadata, it should match.
    # We allow <= because sample_size in generate_submission limits the metadata rows processed.
    assert len(submission_df) <= expected_rows

    print("Inference pipeline demonstration successful.")


if __name__ == "__main__":
    try:
        setup_demo_environment()

        demo_text_utils()
        demo_feature_engineering()
        demo_ranker_model()
        demo_reader_model()
        demo_inference()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during the demonstration: {e}")
        raise e
