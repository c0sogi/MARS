import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from the provided library
from library.config import SUBMISSION_DIR, RANDOM_SEED, INPUT_DIR
from library.utils import set_seed, clipped_log_loss
from library.data_loader import prepare_datasets
from library.preprocessing import preprocess_data
from library.model_factory import create_expert_library
from library.ensemble_selection import select_best_ensemble


def main():
    # 1. Setup and Reproducibility
    print("Initializing...")
    set_seed(RANDOM_SEED)

    # 2. Data Loading
    # Loads metadata, extracts raw features, and generates probabilistic morphometric features.
    # Uses caching to speed up subsequent runs.
    print("Loading datasets...")
    dataset = prepare_datasets(load_cached_data=True)

    # Verification: Check dataset structure
    assert "train" in dataset and "val" in dataset and "test" in dataset
    assert "classes" in dataset
    print(f"Loaded {len(dataset['classes'])} classes.")
    print(f"Training samples: {len(dataset['train']['y'])}")
    print(f"Validation samples: {len(dataset['val']['y'])}")

    # 3. Preprocessing
    # Applies PowerTransformer to standardize distributions.
    print("Preprocessing data...")
    processed_data = preprocess_data(dataset, load_cached_data=True)

    # Verification: Check if preprocessing modified the data (e.g., standardization)
    # We check the 'Global' view of the training set.
    # After PowerTransformer(standardize=True), mean should be approx 0 and std approx 1.
    train_global_view = processed_data["train"]["views"]["Global"]
    assert (
        np.abs(train_global_view.mean()) < 0.1
    ), "Data does not appear to be centered after preprocessing."
    assert (
        0.8 < train_global_view.std() < 1.2
    ), "Data does not appear to be scaled after preprocessing."

    # 4. Model Training & Expert Evaluation
    print("Training expert models...")
    experts = create_expert_library()

    val_preds_dict = {}
    test_preds_dict = {}

    # Get labels
    y_train = processed_data["train"]["y"]
    y_val = processed_data["val"]["y"]

    for model_name, model, view_name in experts:
        # Retrieve the specific view for this expert
        X_train = processed_data["train"]["views"][view_name]
        X_val = processed_data["val"]["views"][view_name]
        X_test = processed_data["test"]["views"][view_name]

        # Train
        model.fit(X_train, y_train)

        # Predict
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)

        # Store predictions
        val_preds_dict[model_name] = val_probs
        test_preds_dict[model_name] = test_probs

        # Optional: Check individual model performance
        loss = clipped_log_loss(y_val, val_probs)
        # print(f"  {model_name} ({view_name}): Log Loss = {loss:.4f}")

    # 5. Ensemble Selection
    # Find the best combination of models using Greedy Forward Selection
    print("Performing ensemble selection...")
    # Limit iterations for speed in this demo
    ensemble_weights = select_best_ensemble(val_preds_dict, y_val, max_iter=20)

    if not ensemble_weights:
        raise RuntimeError("Ensemble selection failed to select any models.")

    # 6. Generate Final Predictions
    print("Generating final submission...")

    # Initialize weighted sum
    # Shape: (n_test_samples, n_classes)
    first_model_preds = list(test_preds_dict.values())[0]
    final_test_preds = np.zeros_like(first_model_preds)
    total_weight = 0

    for model_name, weight in ensemble_weights.items():
        final_test_preds += test_preds_dict[model_name] * weight
        total_weight += weight

    # Normalize
    final_test_preds /= total_weight

    # 7. Format Submission
    # Load test metadata to get IDs
    test_metadata = pd.read_csv("./metadata/test.csv")
    test_ids = test_metadata["id"]

    # Create DataFrame
    submission = pd.DataFrame(final_test_preds, columns=dataset["classes"])
    submission.insert(0, "id", test_ids)

    # Verification: Compare against sample submission format
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    # Check columns
    assert list(submission.columns) == list(
        sample_sub.columns
    ), "Submission columns do not match sample submission."
    # Check rows
    assert len(submission) == len(
        sample_sub
    ), f"Submission row count mismatch ({len(submission)} vs {len(sample_sub)})."
    # Check for NaNs
    assert not submission.isnull().values.any(), "Submission contains NaN values."

    # Save
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Successfully saved submission to {submission_path}")
    print("Done.")


if __name__ == "__main__":
    main()
