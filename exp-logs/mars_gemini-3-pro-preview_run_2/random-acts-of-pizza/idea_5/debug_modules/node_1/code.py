import os
import sys
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample

# Import library modules
from library import config
from library import utils
from library import data_handler
from library import preprocessor
from library import siamese_trainer
from library import feature_extractor
from library import classifier


def main():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print(">>> Step 1: Setup and Configuration")

    # Set seed for reproducibility
    utils.set_seed(42)

    # Override config for speed in this demonstration
    print("Overriding configuration for fast execution...")
    config.FINE_TUNE_EPOCHS = 1  # Reduce epochs to 1
    config.FINE_TUNE_BATCH_SIZE = 8  # Reduce batch size
    config.CV_FOLDS = 2  # Reduce CV folds
    config.CLASSIFIER_MAX_ITER = 100  # Limit solver iterations
    config.CLASSIFIER_C_GRID = [0.1, 1.0]  # Smaller grid for hyperparam search

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print("-" * 30)

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print(">>> Step 2: Data Loading (library.data_handler)")

    # Load datasets from scratch (ignoring cache to prove logic works)
    df_train, df_val, df_test = data_handler.load_datasets(load_cached_data=False)

    # Verification
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    assert not df_train.empty, "Training dataframe is empty."
    assert config.TARGET_COL in df_train.columns, "Target column missing in train."
    assert (
        "combined_text" in df_train.columns
    ), "Text preprocessing failed (combined_text missing)."
    print("Data loading verified.")
    print("-" * 30)

    # ==========================================
    # 3. Preprocessing Demonstration
    # ==========================================
    print(">>> Step 3: Preprocessing Components (library.preprocessor)")

    # Demonstrate Tabular Scaler
    print("Fitting TabularScaler...")
    scaler = preprocessor.TabularScaler()
    scaler.fit(df_train)
    X_train_scaled = scaler.transform(df_train)

    assert X_train_scaled.shape[0] == len(
        df_train
    ), "Scaled features row count mismatch."
    assert X_train_scaled.shape[1] == len(
        config.NUMERICAL_COLS
    ), "Scaled features column count mismatch."

    # Demonstrate Siamese Dataset Builder
    print("Creating Siamese InputExamples...")
    dataset_builder = preprocessor.SiameseDatasetBuilder()
    examples = dataset_builder.create_examples(df_train.head(50))  # Test on subset

    assert len(examples) == 50, "Incorrect number of input examples created."
    assert isinstance(
        examples[0], InputExample
    ), "Examples are not of type InputExample."
    print("Preprocessing components verified.")
    print("-" * 30)

    # ==========================================
    # 4. Siamese Network Training
    # ==========================================
    print(">>> Step 4: Siamese Network Training (library.siamese_trainer)")

    # Instantiate FineTuner
    tuner = siamese_trainer.FineTuner()

    # Train the model
    # We force training (load_cached_data=False) to demonstrate the training loop.
    # Since we set epochs=1 and batch_size=8, this should be relatively fast.
    print("Starting fine-tuning (this may take a moment)...")
    tuner.train(load_cached_data=False)

    # Verify model artifact exists
    assert os.path.exists(
        config.FINE_TUNED_MODEL_PATH
    ), "Fine-tuned model directory not found."

    # Verify encoding capability
    test_sentences = ["I need pizza", "Pizza is great"]
    embeddings = tuner.encode(test_sentences)
    assert embeddings.shape[0] == 2, "Embedding batch size mismatch."
    print("Siamese training and encoding verified.")
    print("-" * 30)

    # ==========================================
    # 5. Feature Extraction
    # ==========================================
    print(">>> Step 5: Feature Extraction (library.feature_extractor)")

    extractor = feature_extractor.FeatureEngineer()

    # Generate features.
    # We set load_cached_data=True.
    # Logic:
    # 1. It checks if X_train_combined.npy exists. If not (likely, unless run before), it proceeds.
    # 2. It calls tuner.train(load_cached_data=True). Since we just trained it in Step 4,
    #    it finds the model on disk and skips re-training.
    # 3. It generates embeddings and concatenates with scaled tabular data.
    X_train, y_train, X_val, y_val, X_test = extractor.generate_features(
        load_cached_data=True
    )

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_test shape:  {X_test.shape}")

    # Expected dimensions: Embedding Size (384 for all-MiniLM-L6-v2) + Numerical Cols
    expected_dim = 384 + len(config.NUMERICAL_COLS)
    assert (
        X_train.shape[1] == expected_dim
    ), f"Feature dimension mismatch. Expected {expected_dim}, got {X_train.shape[1]}"
    assert len(y_train) == len(df_train), "Label count mismatch."
    print("Feature extraction verified.")
    print("-" * 30)

    # ==========================================
    # 6. Classification and Submission
    # ==========================================
    print(">>> Step 6: Classification (library.classifier)")

    clf = classifier.PizzaClassifier()

    # Optimize hyperparameters
    print("Optimizing classifier...")
    clf.optimize(X_train, y_train)

    # Evaluate on Validation set
    print("Evaluating on Validation set...")
    val_auc = clf.evaluate(X_val, y_val)

    assert 0.0 <= val_auc <= 1.0, "AUC score out of bounds."

    # Generate Submission
    print("Generating submission...")
    clf.generate_submission(X_test)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert len(sub_df) == len(df_test), "Submission row count mismatch."
    assert config.ID_COL in sub_df.columns, "Submission ID column missing."
    assert config.TARGET_COL in sub_df.columns, "Submission target column missing."

    print("Classification and submission verified.")
    print("-" * 30)

    print(">>> Execution Completed Successfully.")


if __name__ == "__main__":
    main()
