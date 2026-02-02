import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.data_loader import PizzaDataLoader
from library.feature_extraction import FeatureExtractor
from library.preprocessing import MultiModalTransformer
from library.model import ModelFactory
from library.pipeline import CrossValidationManager


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 0. Setup and Config Override for Demo Speed
    print("Initializing Demo...")
    set_seed(42)

    # Override Config for speed and isolation
    # We use a very small subset and minimal folds to ensure this runs instantly
    Config.DEBUG_SAMPLE_SIZE = 20
    Config.NUM_FOLDS = 2
    Config.N_BAGGING_ESTIMATORS = 2
    Config.PCA_COMPONENTS = 10  # Reduced to accommodate small sample size
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Simplify Grid Search to a single point to avoid time consumption
    Config.LR_PARAM_GRID = {
        "bagging__estimator__C": [1.0],
        "bagging__estimator__class_weight": [None],
    }

    # Clean up demo directories if they exist to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(
        f"Config configured: Sample Size={Config.DEBUG_SAMPLE_SIZE}, Folds={Config.NUM_FOLDS}"
    )

    # 1. Test Data Loader
    print("\n--- Testing Data Loader ---")
    loader = PizzaDataLoader()

    # Load train data
    # This triggers the loading of metadata and raw JSON, merging them, and sampling
    df_train = loader.load_data("train", load_cached_data=False)
    print(f"Loaded Train Data Shape: {df_train.shape}")

    # Assertions
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(df_train)}"
    assert "request_id" in df_train.columns
    assert "requester_received_pizza" in df_train.columns

    # Load metadata features
    meta_features = loader.get_metadata_features(df_train)
    print(f"Metadata Features Shape: {meta_features.shape}")
    assert len(meta_features) == len(df_train)
    assert (
        meta_features.select_dtypes(include=[np.number]).shape[1]
        == meta_features.shape[1]
    ), "Metadata should be all numeric"

    # 2. Test Feature Extraction
    print("\n--- Testing Feature Extraction ---")
    extractor = FeatureExtractor()

    # We force re-computation to verify the logic, bypassing cache for the demo
    features = extractor.extract_features(df_train, "train", load_cached_data=False)

    print("Extracted Feature Keys:", features.keys())

    # Assertions
    assert "anchor" in features
    assert "semantic_aux" in features
    assert "affective_aux" in features

    # Check shapes
    # Anchor: MiniLM-L6-v2 -> 384 dim
    assert features["anchor"].shape == (Config.DEBUG_SAMPLE_SIZE, 384)
    # Semantic Aux: mpnet-base-v2 -> 768 dim
    assert features["semantic_aux"].shape == (Config.DEBUG_SAMPLE_SIZE, 768)
    # Affective Aux: roberta-base-go_emotions -> 28 dim (28 output logits)
    assert features["affective_aux"].shape == (Config.DEBUG_SAMPLE_SIZE, 28)

    # 3. Test Preprocessing (MultiModalTransformer)
    print("\n--- Testing Preprocessing ---")
    # Prepare input dictionary for transformer
    X_dict = {
        "anchor": features["anchor"],
        "semantic_aux": features["semantic_aux"],
        "affective_aux": features["affective_aux"],
        "metadata": meta_features,
    }

    transformer = MultiModalTransformer()

    # Fit
    transformer.fit(X_dict)
    print("Transformer fitted successfully.")

    # Transform
    X_fused = transformer.transform(X_dict)
    print(f"Fused Feature Shape: {X_fused.shape}")

    # Validate dimensions
    # View 1 (Anchor): 384
    # View 2 (Semantic Aux): PCA components (Config.PCA_COMPONENTS = 10)
    # View 3 (Affective Aux): 28
    # View 4 (Metadata): 10 (based on get_metadata_features list)
    expected_dim = 384 + Config.PCA_COMPONENTS + 28 + 10
    assert X_fused.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        expected_dim,
    ), f"Expected dim {expected_dim}, got {X_fused.shape[1]}"

    # 4. Test Model Factory
    print("\n--- Testing Model Factory ---")
    clf = ModelFactory.get_classifier()
    print(f"Classifier Type: {type(clf)}")
    assert hasattr(clf, "fit") and hasattr(
        clf, "predict"
    ), "Classifier must comply with sklearn API"

    grid = ModelFactory.get_hyperparameter_grid()
    print(f"Hyperparameter Grid: {grid}")
    assert isinstance(grid, dict)

    # 5. Test Pipeline Execution (CrossValidationManager)
    print("\n--- Testing CrossValidationManager (Full Pipeline) ---")
    manager = CrossValidationManager()

    # Run CV
    # This will internally load data, extract features, run grid search, and train models.
    # We rely on the Config overrides to make this fast.
    # load_cached_data=True allows it to pick up the features we extracted in step 2 if names match,
    # or recompute if necessary.
    manager.run_cv(load_cached_data=True)

    # Check if models were saved
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_pipeline.joblib")
        assert os.path.exists(
            model_path
        ), f"Model for fold {fold} not found at {model_path}"

    # Generate Submission
    manager.generate_submission(load_cached_data=False)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not generated"

    # Validate Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), "Submission should have rows equal to debug sample size"
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns
    assert df_sub["requester_received_pizza"].dtype == float

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
