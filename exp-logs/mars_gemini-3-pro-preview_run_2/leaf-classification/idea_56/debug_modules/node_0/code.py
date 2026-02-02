import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import provided library components
from library.utils import set_seed, create_submission
from library.data_factory import DataFactory
from library.pipeline_definitions import get_expert_library
from library.ensemble_strategy import GreedySelector


def main():
    # 1. Setup
    print("Initializing demonstration...")
    set_seed(42)

    # Define paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2. Data Loading and Feature Extraction
    print("\nLoading datasets and extracting features...")
    # Instantiate DataFactory with specific cache dir to demonstrate caching mechanism
    factory = DataFactory(metadata_dir=METADATA_DIR, cache_dir=CACHE_DIR)

    # Load datasets (Train, Val, Test)
    # This triggers image feature extraction (morphometrics) and merging with tabular data
    df_train, df_val, df_test = factory.load_datasets(load_cached_data=False)

    # Verification: Check dimensions and new features
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    # Verify Morphometric features exist (e.g., 'hu_0', 'solidity')
    assert "hu_0" in df_train.columns, "Morphometric feature 'hu_0' missing."
    assert "solidity" in df_train.columns, "Morphometric feature 'solidity' missing."
    assert "margin_1" in df_train.columns, "Original feature 'margin_1' missing."

    # 3. Data Preparation for Modeling
    print("\nPreparing data for modeling...")
    target_col = "species"

    # Encode Labels
    le = LabelEncoder()
    y_train = le.fit_transform(df_train[target_col])
    y_val = le.transform(df_val[target_col])
    class_names = list(le.classes_)

    # Get Feature Groups for the pipelines
    # The pipelines need to know which columns belong to 'shape', 'margin', etc.
    feature_groups = factory.get_feature_groups(df_train)

    # 4. Model Training (Expert Pipelines)
    print("\nTraining expert pipelines...")
    # Get the library of pipelines defined in pipeline_definitions.py
    # We use a small shrinkage list for speed in this demo
    experts = get_expert_library(feature_groups, shrinkage_levels=[0.1, 0.5])

    val_preds = {}
    test_preds = {}

    # Train each expert
    for name, pipeline in experts.items():
        # Fit on training data
        # Note: We pass the full dataframe; ColumnTransformers in the pipeline select specific columns
        pipeline.fit(df_train, y_train)

        # Predict probabilities
        p_val = pipeline.predict_proba(df_val)
        p_test = pipeline.predict_proba(df_test)

        val_preds[name] = p_val
        test_preds[name] = p_test

    print(f"Trained {len(experts)} experts.")

    # 5. Ensemble Selection
    print("\nOptimizing ensemble weights (Greedy Selection)...")
    selector = GreedySelector(max_iterations=5, tolerance=1e-5, verbose=True)

    # Fit selector on validation data
    # We pass the list of integer labels that the model was trained on
    selector.fit(val_preds, y_val, labels=list(range(len(class_names))))

    # Verification: Ensure weights were found
    assert len(selector.weights_) > 0, "GreedySelector failed to select any experts."

    # 6. Final Prediction
    print("\nGenerating final predictions...")
    final_test_probs = selector.predict(test_preds)

    # Verification: Check probability shape and range
    assert final_test_probs.shape == (len(df_test), len(class_names))
    assert np.all(final_test_probs >= 0) and np.all(final_test_probs <= 1 + 1e-9)

    # 7. Create Submission
    print("\nCreating submission file...")
    create_submission(
        ids=df_test["id"].values,
        predictions=final_test_probs,
        class_names=class_names,
        output_path=SUBMISSION_PATH,
    )

    # Final check
    if os.path.exists(SUBMISSION_PATH):
        print(f"Success! Submission generated at {SUBMISSION_PATH}")
        # Print first few lines to verify format
        with open(SUBMISSION_PATH, "r") as f:
            print("First 3 lines of submission:")
            for _ in range(3):
                print(f.readline().strip())
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
