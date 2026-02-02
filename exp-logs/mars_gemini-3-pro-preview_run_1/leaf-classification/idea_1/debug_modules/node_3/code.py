import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import load_data
from library.model import LeafModel


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Configure for Speed and Demo purposes
    # We modify the Config class attributes directly to control the behavior of the library modules
    print("Configuring parameters for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Use only 50 samples for the demo
    Config.NUM_BOOST_ROUND = 10  # Train for only 10 iterations
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = 0  # Suppress evaluation logging
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir for demo
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Setup directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.RANDOM_SEED)
    print("Configuration complete.")

    # 2. Demonstrate Data Loading
    print("\n--- Demonstrating Data Loading ---")
    # We force reload (load_cached_data=False) to show the processing logic
    # We use debug=True to get the small subset defined above
    X, y, X_test, test_ids, label_encoder = load_data(
        load_cached_data=False, debug=True
    )

    # Validation of Loaded Data
    print("Validating loaded data shapes...")
    assert (
        len(X) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} training samples, got {len(X)}"
    assert (
        len(y) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} labels, got {len(y)}"
    assert (
        len(X_test) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} test samples (debug mode), got {len(X_test)}"
    assert len(test_ids) == Config.DEBUG_SAMPLES, "Test IDs count mismatch"
    assert hasattr(label_encoder, "classes_"), "LabelEncoder is not fitted"

    print(f"Data Loaded Successfully.")
    print(f"Training Data Shape: {X.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(label_encoder.classes_)}")

    # 3. Demonstrate Model Training
    print("\n--- Demonstrating Model Training ---")

    # Create a simple split for training demonstration
    # We disable stratification here because the small debug sample size (50)
    # with 99 classes leads to classes with single instances, causing errors.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=None
    )

    # Initialize Model
    model = LeafModel()

    # Train Model
    print("Training LightGBM model...")
    model.train(X_train, y_train, X_val, y_val)

    # Validate Model State
    assert model.model is not None, "Model should be initialized after training"
    print("Model training complete.")

    # 4. Demonstrate Prediction
    print("\n--- Demonstrating Prediction ---")

    # Predict on Test Set
    probabilities = model.predict(X_test)

    # Handle schema mismatch caused by class filtering in debug mode (Cite debug_lesson_2)
    # If the model was trained on a subset of classes, project predictions back to the full schema.
    if probabilities.shape[1] != len(label_encoder.classes_):
        print(
            f"Debug: Projecting predictions from shape {probabilities.shape} to full schema."
        )
        full_probs = np.zeros((len(probabilities), len(label_encoder.classes_)))
        # model.model.classes_ contains the class indices seen during training
        full_probs[:, model.model.classes_] = probabilities
        probabilities = full_probs

    # Validate Predictions
    assert probabilities.shape == (
        len(X_test),
        len(label_encoder.classes_),
    ), f"Prediction shape mismatch. Expected {(len(X_test), len(label_encoder.classes_))}, got {probabilities.shape}"

    # Check probability constraints (0 <= p <= 1)
    assert np.all(probabilities >= 0) and np.all(
        probabilities <= 1
    ), "Probabilities out of range [0, 1]"

    print("Predictions generated successfully.")

    # 5. Demonstrate Submission Generation
    print("\n--- Demonstrating Submission Generation ---")

    save_submission(
        predictions=probabilities,
        test_ids=test_ids,
        label_encoder=label_encoder,
        output_path=Config.SUBMISSION_FILE,
    )

    # Validate Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check dimensions: rows = samples, cols = id + classes
    expected_cols = 1 + len(label_encoder.classes_)  # id + species
    assert (
        df_sub.shape[1] == 100
    ), f"Expected 100 columns (from sample_submission), got {df_sub.shape[1]}"
    assert (
        df_sub.shape[0] == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} rows, got {df_sub.shape[0]}"
    assert Config.ID_COL in df_sub.columns, f"Missing ID column '{Config.ID_COL}'"

    # Check that values are clipped (none should be exactly 0 or 1 if epsilon is applied correctly,
    # though strict 0/1 might exist if the model is very confident and epsilon is small.
    # The utility clips to [epsilon, 1-epsilon]).
    epsilon = Config.PROB_CLIP_EPSILON
    feature_cols = [c for c in df_sub.columns if c != Config.ID_COL]
    min_val = df_sub[feature_cols].min().min()
    max_val = df_sub[feature_cols].max().max()

    assert min_val >= epsilon, f"Found probability lower than epsilon: {min_val}"
    assert max_val <= (
        1.0 - epsilon
    ), f"Found probability higher than 1-epsilon: {max_val}"

    print(f"Submission file validated at: {Config.SUBMISSION_FILE}")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
