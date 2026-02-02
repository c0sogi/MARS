import sys
import os
import shutil
import numpy as np
import pandas as pd
import random
import lightgbm as lgb

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Import library modules
from library.config import PathConfig, ModelConfig, FeatureConfig
from library.text_processing import TextPreprocessor
from library.corpus_stats import IDFIndex
from library.feature_extractor import CandidateFeatureGenerator
from library.data_loader import NQDataReader
from library.ranker_model import GradientBoostingRanker
from library.answer_selector import ShortAnswerHeuristic


def run_demonstration():
    print("--- Starting Library Demonstration ---")

    # 1. Setup / Monkey-patching for Speed and Isolation
    # We redirect working files to a demo folder to avoid messing with real training artifacts
    # and to ensure we actually run the computation logic on small data.
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Patch paths in PathConfig to point to our demo directory
    PathConfig.WORKING_DIR = demo_dir
    PathConfig.IDF_CACHE = os.path.join(demo_dir, "idf_stats.npy")
    PathConfig.TRAIN_FEATURES_CACHE = os.path.join(demo_dir, "train_features.parquet")
    PathConfig.VAL_FEATURES_CACHE = os.path.join(demo_dir, "val_features.parquet")
    PathConfig.TEST_FEATURES_CACHE = os.path.join(demo_dir, "test_features.parquet")
    PathConfig.MODEL_FILE = os.path.join(demo_dir, "lgbm_ranker.txt")

    # Patch Model Config for speed
    ModelConfig.NUM_BOOST_ROUND = 10
    ModelConfig.EARLY_STOPPING_ROUNDS = 5
    # Ensure verbose is off to keep output clean
    ModelConfig.LGBM_PARAMS["verbose"] = -1
    ModelConfig.LGBM_PARAMS["seed"] = 42

    print("Configuration updated for fast demonstration.")

    # 2. Text Preprocessing Verification
    print("\n--- Testing TextPreprocessor ---")
    preprocessor = TextPreprocessor()
    text = "The quick brown foxes are jumping over fences!"
    tokens = preprocessor.preprocess(text)
    print(f"Original: '{text}'")
    print(f"Processed: {tokens}")

    # Assertions
    # 'the', 'are', 'over' are stopwords (default config).
    # 'foxes' -> 'fox', 'jumping' -> 'jump', 'fences' -> 'fenc' (Porter Stemmer behavior)
    assert isinstance(tokens, list), "Output must be a list"
    assert len(tokens) > 0, "Should return tokens"
    assert "the" not in tokens, "Stopwords should be removed"

    if FeatureConfig.USE_STEMMING:
        # Check if stemming occurred (foxes -> fox/foxes depending on stemmer logic, usually 'fox')
        # We check that the original plural form is likely gone or changed
        assert "foxes" not in tokens, "Stemming should reduce 'foxes'"

    print("TextPreprocessor passed.")

    # 3. Corpus Stats (IDF) Verification
    print("\n--- Testing IDFIndex ---")
    # We force build_from_corpus with a small sample size.
    # load_cached_data=False ensures we compute it now instead of looking for non-existent cache.
    idf_index = IDFIndex()
    # Note: The class modifies the extension to .parquet internally
    idf_index.build_from_corpus(sample_size=100, load_cached_data=False)

    # Check IDF values
    assert len(idf_index.idf_map) > 0, "IDF map should not be empty after processing"

    # Check consistency
    test_token = list(idf_index.idf_map.keys())[0]
    val = idf_index.get_idf(test_token)
    assert val > 0, "IDF value should be positive"

    # Check OOV behavior
    oov_val = idf_index.get_idf("supercalifragilisticexpialidocious_unknown_token")
    assert oov_val == idf_index.default_idf, "OOV should return default IDF"
    print("IDFIndex passed.")

    # 4. Feature Extraction Verification
    print("\n--- Testing CandidateFeatureGenerator ---")
    # Pass the already built idf_index to save time
    feature_gen = CandidateFeatureGenerator(idf_index=idf_index)

    q_text = "What is the capital of France?"
    c_text = "Paris is the capital of France."
    # idx 0, total 10 candidates
    features = feature_gen.extract_features(q_text, c_text, 0, 10)

    print(f"Extracted feature shape: {features.shape}")
    print(f"Feature vector: {features}")

    # We expect: BM25(1) + TFIDF(1) + Lexical(3) + Positional(3) + Length(1) = 9 features
    expected_dim = 9
    assert features.shape == (
        expected_dim,
    ), f"Expected {expected_dim} features, got {features.shape[0]}"
    assert not np.isnan(features).any(), "Features should not contain NaNs"
    print("CandidateFeatureGenerator passed.")

    # 5. Data Loading Verification
    print("\n--- Testing NQDataReader ---")
    data_loader = NQDataReader()

    # Load small training set
    # We need to ensure we don't rely on existing cache in ./working/idea_2,
    # but we patched PathConfig.TRAIN_FEATURES_CACHE to ./working/demo_run/...
    # sample_size=50 to be fast
    print("Loading training samples...")
    train_df = data_loader.get_training_samples(sample_size=50, load_cached_data=False)

    assert not train_df.empty, "Training dataframe should not be empty"
    assert "label" in train_df.columns, "Training data must have labels"
    assert "f_0" in train_df.columns, "Training data must have feature columns"
    print(f"Loaded {len(train_df)} training samples.")

    # Load small validation set
    print("Loading validation samples...")
    val_df = data_loader.get_validation_samples(sample_size=20, load_cached_data=False)
    assert not val_df.empty, "Validation dataframe should not be empty"
    print(f"Loaded {len(val_df)} validation samples.")
    print("NQDataReader passed.")

    # 6. Model Training Verification
    print("\n--- Testing GradientBoostingRanker ---")
    ranker = GradientBoostingRanker()

    # Train
    print("Training model...")
    ranker.train_model(train_df, val_df)

    # Check if model file created
    assert os.path.exists(PathConfig.MODEL_FILE), "Model file was not saved"

    # Predict
    print("Predicting on validation set...")
    scores = ranker.predict_scores(val_df)
    assert len(scores) == len(val_df), "Number of scores must match number of samples"
    assert np.all(
        (scores >= 0) & (scores <= 1)
    ), "Scores must be probabilities between 0 and 1"
    print(f"Prediction mean: {np.mean(scores):.4f}")
    print("GradientBoostingRanker passed.")

    # 7. Answer Selection Verification
    print("\n--- Testing ShortAnswerHeuristic ---")
    heuristic = ShortAnswerHeuristic()

    # Test Sentence Selection
    # We construct a case where one sentence clearly overlaps more
    q_sel = "Who wrote Harry Potter?"
    long_ans = (
        "J.K. Rowling wrote the Harry Potter series. It is a very popular series."
    )

    best_sent, score = heuristic.find_best_sentence(q_sel, long_ans)
    print(f"Question: {q_sel}")
    print(f"Long Answer: {long_ans}")
    print(f"Selected: '{best_sent}' (Score: {score:.4f})")

    # Depending on stemming/stop words, "J.K. Rowling wrote the Harry Potter series" should match "Who wrote Harry Potter"
    # Overlap: wrote, harry, potter.
    # The other sentence: "It is a very popular series" -> series.
    # First sentence should win.
    assert best_sent is not None, "Should find a sentence"
    # Check if the selected sentence contains key information
    assert "Rowling" in best_sent, "Should select the sentence containing the answer"

    # Test Yes/No
    yes_text = "Yes, he did it."
    no_text = "No, that is incorrect."
    none_text = "Maybe, who knows."

    assert heuristic.check_yes_no(yes_text) == "YES", "Should detect YES"
    assert heuristic.check_yes_no(no_text) == "NO", "Should detect NO"
    assert heuristic.check_yes_no(none_text) == "NONE", "Should return NONE"
    print("ShortAnswerHeuristic passed.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
