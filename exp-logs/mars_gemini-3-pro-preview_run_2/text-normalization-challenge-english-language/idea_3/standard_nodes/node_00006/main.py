import pandas as pd
import numpy as np
import os
import random
import time
import sys

# Import from the provided library
from library.config import SEED, SUBMISSION_FILE
from library.data_loader import load_val_data, load_test_data
from library.features import (
    load_or_create_train_features,
    create_test_features,
    SubwordEmbedder,
    EMBEDDER_PATH,
)
from library.retrieval import load_or_train_index
from library.normalizers import NormalizationRegistry


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    print("Starting runfile.py execution...")
    start_time = time.time()

    # 2. Load Training Data & Features
    # This utilizes the caching mechanism in library/features.py
    print("Loading training features...")
    train_vectors, train_labels, embedder = load_or_create_train_features(
        load_cached_data=True
    )

    # Ensure embedder is loaded (in case of cache partial hit where vectors exist but embedder obj is None)
    if embedder is None:
        if os.path.exists(EMBEDDER_PATH):
            print(f"Loading embedder from {EMBEDDER_PATH}...")
            embedder = SubwordEmbedder.load(EMBEDDER_PATH)
        else:
            raise FileNotFoundError(
                "Embedder model not found. Training features might be corrupted or incomplete."
            )

    # 3. Limit Training Data for Speed (Baseline Requirement)
    # The full dataset (even after PLAIN downsampling) might be too large for k-NN inference
    # on CPU within the time limit. We cap the training set size to ensure fast execution.
    MAX_TRAIN_SAMPLES = 50000
    if len(train_vectors) > MAX_TRAIN_SAMPLES:
        print(
            f"Downsampling training set from {len(train_vectors)} to {MAX_TRAIN_SAMPLES} for fast baseline..."
        )
        indices = np.arange(len(train_vectors))
        np.random.shuffle(indices)
        selected_indices = indices[:MAX_TRAIN_SAMPLES]
        train_vectors = train_vectors[selected_indices]
        train_labels = train_labels[selected_indices]

    # 4. Train k-NN Model
    # We force training a new index (load_cached_model=False) because we might have
    # dynamically subsampled the data above.
    print("Building k-NN index...")
    classifier = load_or_train_index(
        train_vectors, train_labels, load_cached_model=False
    )

    # 5. Validation
    print("Loading validation data...")
    val_df = load_val_data(load_cached_data=True)

    # Transform validation features
    print("Transforming validation features...")
    val_vectors = embedder.transform(val_df)

    print("Predicting validation classes...")
    val_pred_classes = classifier.predict(val_vectors)

    print("Applying normalization rules...")
    registry = NormalizationRegistry()

    # Extract arrays for fast iteration
    val_before = val_df["before"].astype(str).values
    val_after_true = val_df["after"].astype(str).values

    # Apply normalization logic based on predicted class
    val_pred_strings = []
    for text, cls in zip(val_before, val_pred_classes):
        val_pred_strings.append(registry.normalize(text, cls))
    val_pred_strings = np.array(val_pred_strings)

    # Calculate Metric (Exact String Match)
    accuracy = np.mean(val_pred_strings == val_after_true)
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("Performing failure analysis...")
    errors = (val_pred_strings != val_after_true).astype(int)
    lengths = np.array([len(t) for t in val_before])

    if len(errors) > 0 and np.std(errors) > 0 and np.std(lengths) > 0:
        corr = np.corrcoef(errors, lengths)[0, 1]
        print(f"Correlation between Error and Token Length: {corr}")
    else:
        print("Correlation between Error and Token Length: NaN (No variance)")

    # 6. Submission
    print("Generating submission...")
    test_df = load_test_data(load_cached_data=True)

    print("Transforming test features...")
    test_vectors = create_test_features(embedder, load_cached_data=True)

    print("Predicting test classes...")
    test_pred_classes = classifier.predict(test_vectors)

    print("Applying normalization to test set...")
    test_before = test_df["before"].astype(str).values
    test_ids = test_df["id"].values

    test_pred_strings = []
    for text, cls in zip(test_before, test_pred_classes):
        test_pred_strings.append(registry.normalize(text, cls))

    submission_df = pd.DataFrame({"id": test_ids, "after": test_pred_strings})

    print(f"Saving submission to {SUBMISSION_FILE}...")
    submission_df.to_csv(SUBMISSION_FILE, index=False)

    elapsed = time.time() - start_time
    print(f"Execution completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
