import os
import sys
import numpy as np
import pandas as pd
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library import config
from library import utils
from library import data_loader
from library import preprocessing
from library import model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_config():
    print("\n=== 1. Demonstrating Configuration ===")
    print(f"Input Directory: {config.INPUT_DIR}")
    print(f"Metadata Directory: {config.METADATA_DIR}")
    print(f"Number of Features: {len(config.FEATURES)}")

    # Assertions
    assert len(config.FEATURES) == 192, "Feature count mismatch in config."
    assert config.TARGET_COL == "species", "Target column mismatch."
    print("Configuration verified.")


def demo_data_loader():
    print("\n=== 2. Demonstrating Data Loader ===")

    # Load data (forcing reload to ignore cache for demonstration)
    print("Loading dataset (ignoring cache)...")
    (train_data, val_data, test_data) = data_loader.load_dataset(load_cached_data=False)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}, ids={ids_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}, ids={ids_val.shape}")
    print(f"Test shapes:  X={X_test.shape}, ids={ids_test.shape}")

    # Assertions
    assert X_train.shape[1] == 192, "Incorrect number of features in training set."
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length."
    assert len(X_val) > 0, "Validation set is empty."
    print("Data loading verified.")

    return train_data, val_data


def demo_preprocessing(train_data):
    print("\n=== 3. Demonstrating Preprocessing (IterativeGaussianizer) ===")
    X_train_raw, y_train, _ = train_data

    # Instantiate the transformer
    transformer = preprocessing.IterativeGaussianizer()

    # Fit on a subset for speed demonstration
    # Increase subset size to > 192 (n_features) to prevent PCA dimensionality reduction
    subset_size = min(300, len(X_train_raw))
    X_subset = X_train_raw.iloc[:subset_size].copy()

    print(f"Fitting transformer on subset of size {X_subset.shape}...")
    transformer.fit(X_subset)

    print("Transforming subset...")
    X_transformed = transformer.transform(X_subset)

    # Assertions
    assert X_transformed.shape == X_subset.shape, "Transformed shape mismatch."
    assert (
        X_transformed.dtype == config.FLOAT_PRECISION
    ), "Incorrect precision after transform."

    # Check if data is roughly centered (standardized)
    means = np.mean(X_transformed, axis=0)
    print(f"Mean of means after transform (should be close to 0): {np.mean(means):.4f}")
    assert np.abs(np.mean(means)) < 0.5, "Transformed data is not centered."

    print("Preprocessing verified.")
    return transformer


def demo_model(train_data, val_data):
    print("\n=== 4. Demonstrating Model (OASDiscriminant) ===")

    # Unpack data
    X_train_raw, y_train, _ = train_data
    X_val_raw, y_val, _ = val_data

    # We need to preprocess the data first using the full pipeline logic
    # or manually use the transformer. Let's use the provided helper
    # to get fully processed data to ensure we match the model's expectations.
    print("Getting fully preprocessed data via helper...")
    (train_proc, val_proc, _) = preprocessing.get_preprocessed_data(
        load_cached_data=False
    )
    X_train, y_train, _ = train_proc
    X_val, y_val, _ = val_proc

    # Initialize Model
    clf = model.OASDiscriminant()

    # Fit Model
    print("Fitting OASDiscriminant...")
    clf.fit(X_train, y_train)

    # Predict
    print("Predicting probabilities on validation set...")
    probs = clf.predict_proba(X_val)

    # Assertions
    n_classes = len(np.unique(y_train))
    assert probs.shape == (
        len(X_val),
        n_classes,
    ), f"Probability shape mismatch. Expected {(len(X_val), n_classes)}, got {probs.shape}"

    # Check row sums (should be 1.0)
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    print("Model training and prediction verified.")
    return clf, probs, y_val


def demo_utils(y_true, y_pred, classes):
    print("\n=== 5. Demonstrating Utilities ===")

    # Compute Log Loss
    loss = utils.compute_log_loss(y_true, y_pred, classes)
    print(f"Computed Log Loss: {loss:.4f}")

    # Assertions
    assert loss > 0, "Log loss should be positive."

    # Save Submission Demo
    dummy_ids = np.arange(len(y_pred))
    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    utils.save_submission(dummy_ids, y_pred, classes, filename=demo_sub_path)

    assert os.path.exists(demo_sub_path), "Submission file was not created."

    # Verify file content format
    df_sub = pd.read_csv(demo_sub_path)
    assert config.ID_COL in df_sub.columns, "ID column missing in submission."
    assert len(df_sub) == len(y_pred), "Row count mismatch in submission."
    assert (
        len(df_sub.columns) == len(classes) + 1
    ), "Column count mismatch (classes + id)."

    print(f"Utilities verified. Demo submission saved to {demo_sub_path}")


def demo_full_pipeline():
    print("\n=== 6. Demonstrating Full Pipeline Execution ===")

    # Run the main orchestrator function
    # This handles loading, preprocessing, training, predicting, and saving.
    model.train_and_predict(load_cached_data=False)

    # Verify final submission exists
    assert os.path.exists(config.SUBMISSION_PATH), "Final submission file missing."
    print(f"Full pipeline completed. Submission saved to {config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(config.SEED)

    # 1. Config
    demo_config()

    # 2. Data Loader
    train_data, val_data = demo_data_loader()

    # 3. Preprocessing
    # We pass the raw data loaded in step 2
    demo_preprocessing(train_data)

    # 4. Model
    # This step internally re-loads/processes data to ensure consistency
    clf, val_probs, val_y = demo_model(train_data, val_data)

    # 5. Utils
    demo_utils(val_y, val_probs, clf.classes_)

    # 6. Full Pipeline
    demo_full_pipeline()

    print("\nAll demonstrations passed successfully.")
