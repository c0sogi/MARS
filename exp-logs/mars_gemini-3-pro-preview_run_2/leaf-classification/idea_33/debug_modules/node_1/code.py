import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library import config
from library import data_loader
from library import feature_extraction
from library import model_library
from library import ensemble_selection


def run_demo():
    # 1. Setup & Configuration
    # ==========================================================================
    print("Initializing Demo...")
    warnings.filterwarnings("ignore")
    np.random.seed(config.RANDOM_STATE)

    # Ensure submission directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading & Validation
    # ==========================================================================
    print("Loading Data Splits...")
    # We use load_cached_data=True to speed up if cache exists,
    # but the logic handles extraction if not.
    X_train, y_train, X_val, y_val = data_loader.get_data_splits(load_cached_data=True)

    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Validation Data Shape: {X_val.shape}")

    # Validation: Ensure we have data and features
    assert not X_train.empty, "Training data is empty."
    assert not X_val.empty, "Validation data is empty."
    assert len(y_train) == len(X_train), "Mismatch in training labels."

    # Check if Macro features are present (columns starting with 'macro' or specific names)
    macro_cols = [c for c in X_train.columns if c in config.MACRO_FEATURE_NAMES]
    assert len(macro_cols) == 11, f"Expected 11 macro features, found {len(macro_cols)}"

    # Encode labels for consistency across models
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = le.classes_
    print(f"  Number of Classes: {len(classes)}")

    # 3. Feature Extraction Logic Verification
    # ==========================================================================
    print("Verifying Feature Extraction Logic...")
    # Pick the first image from metadata to test the extraction function directly
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_image_path = train_meta.iloc[0]["image_path"]

    # Test single image processing
    extracted_feats = feature_extraction.process_single_image(sample_image_path)

    # Validation: Check output shape and type
    assert isinstance(
        extracted_feats, np.ndarray
    ), "Feature extraction should return numpy array"
    assert extracted_feats.shape == (
        11,
    ), f"Expected 11 features, got {extracted_feats.shape}"
    # Check for NaN
    assert not np.isnan(extracted_feats).any(), "Extracted features contain NaNs"
    print("  Feature extraction verification passed.")

    # 4. Model Library Training
    # ==========================================================================
    print("Building and Training Expert Library...")
    experts = model_library.build_expert_library()
    print(f"  Found {len(experts)} expert configurations.")

    trained_pipelines = {}
    val_preds_dict = {}

    # Train each expert
    for i, expert_cfg in enumerate(experts):
        name = expert_cfg["name"]
        pipeline = expert_cfg["pipeline"]

        # Fit on training data
        # Note: The pipeline's FeatureSelector handles selecting the correct columns (Global vs Macro)
        pipeline.fit(X_train, y_train_enc)

        # Store trained pipeline
        trained_pipelines[name] = pipeline

        # Predict on validation set
        # predict_proba returns (n_samples, n_classes)
        preds = pipeline.predict_proba(X_val)
        val_preds_dict[name] = preds

    print("  All experts trained successfully.")

    # 5. Ensemble Selection
    # ==========================================================================
    print("Running Ensemble Selection (Greedy Forward Selection)...")

    # Initialize selector with a reasonable number of steps for the demo
    selector = ensemble_selection.GreedySelector(max_steps=20, tolerance=1e-6)

    # Fit selector
    # We pass y_val_enc (integers) but the selector uses log_loss which handles it
    # provided we pass the class labels to the metric if needed, or we can pass one-hot.
    # However, sklearn log_loss supports integer labels if 'labels' arg is provided.
    # The provided GreedySelector._score method uses self.classes_.
    # We need to pass the class list matching the probability columns.
    # Since we trained on y_train_enc (0..98), the pipeline outputs probs for 0..98.
    # We pass the integer classes [0, 1, ... 98] to the selector.

    selector_classes = np.arange(len(classes))
    selector.fit(val_preds_dict, y_val_enc, selector_classes)

    print(f"  Selected Experts: {selector.selected_experts}")

    # Validation: Ensure at least one expert was selected
    assert len(selector.selected_experts) > 0, "No experts were selected!"

    # 6. Inference on Test Set
    # ==========================================================================
    print("Performing Inference on Test Set...")
    X_test, test_ids = data_loader.get_test_data(load_cached_data=True)

    test_preds_dict = {}

    # Generate predictions for test set using all trained experts
    for name, pipeline in trained_pipelines.items():
        # Only predict if the expert is actually used (optimization)
        if name in selector.selected_experts:
            test_preds_dict[name] = pipeline.predict_proba(X_test)

    # Aggregate predictions using the selector
    final_probs = selector.predict(test_preds_dict)

    print(f"  Final Predictions Shape: {final_probs.shape}")
    assert final_probs.shape == (len(X_test), len(classes)), "Output shape mismatch"

    # 7. Submission Generation
    # ==========================================================================
    print("Generating Submission File...")

    # Construct DataFrame
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"  Submission saved to {config.SUBMISSION_PATH}")

    # Final check
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Peek at the file
    df_check = pd.read_csv(config.SUBMISSION_PATH)
    print(f"  Submission Head:\n{df_check.head(2)}")
    print("Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
