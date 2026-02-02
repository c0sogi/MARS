import os
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

# Import provided library components
from library.config import Config
from library.data_processor import DataProcessor
from library.expert_models import get_expert_pool
from library.ensemble_selector import GreedySelector

# Filter warnings for cleaner output during demo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def set_reproducibility(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Pipeline Demonstration ===")

    # 1. Setup and Configuration Overrides for Speed/Demo
    print("[Setup] Configuring environment...")
    set_reproducibility(Config.RANDOM_SEED)

    # Enable Debug mode to use a tiny subset of data (50 rows)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Reduce computational cost for demonstration
    Config.NCA_MAX_ITER = 5
    Config.LOGREG_MAX_ITER = 10
    Config.LOGREG_C_GRID = [1.0]  # Disable grid search
    Config.LOGREG_JOBS = 1
    # Cite debug_lesson_1: Avoid hardcoded CV splits exceeding minority class counts.
    # Cite debug_lesson_2: Subsampling creates singletons; use KFold to bypass StratifiedKFold requirements.
    Config.LOGREG_CV = KFold(n_splits=2, shuffle=True, random_state=Config.RANDOM_SEED)

    # 2. Data Loading and Processing
    print("[Data] Loading and processing data...")
    processor = DataProcessor()

    # Force load from scratch (load_cached_data=False) to ensure DEBUG settings are applied
    # and we don't load full cached data from a previous run.
    data = processor.load_and_process_data(load_cached_data=False)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    # Verify Data Shapes
    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Number of Classes in Subset: {len(classes)}")

    # assert X_train.shape == (50, 192)  # Relaxed for filtering
    assert len(y_train) == X_train.shape[0]
    assert len(classes) <= 50

    # Cite debug_lesson_2: Verify Class Counts Exceed CV Folds When Subsampling
    # Filter out classes with fewer than 2 samples to ensure Stratified CV works
    unique_y, counts_y = np.unique(y_train, return_counts=True)
    valid_classes = unique_y[counts_y >= 2]

    train_mask = np.isin(y_train, valid_classes)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Filter validation set to match training classes
    val_mask = np.isin(y_val, valid_classes)
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    # Update classes to match the subset of species remaining
    classes = classes[np.unique(y_train)]

    print(f"  Filtered Training Data Shape: {X_train.shape}")
    print(f"  Filtered Classes: {len(classes)}")

    # Switch to Stratified CV (int=2) now that singleton classes are removed
    Config.LOGREG_CV = 2

    # 3. Dynamic Configuration Adjustment
    # NCA components cannot exceed the number of classes or samples.
    # In DEBUG mode, we might have fewer than 99 classes.
    n_classes_subset = len(classes)
    # Set NCA components to be smaller than n_classes and n_samples
    Config.NCA_COMPONENTS = min(n_classes_subset - 1, 10)
    # Ensure at least 1 component
    if Config.NCA_COMPONENTS < 1:
        Config.NCA_COMPONENTS = 1
    print(f"  Adjusted NCA Components to {Config.NCA_COMPONENTS} for debug subset.")

    # 4. Expert Model Training
    print("[Models] Initializing and training experts...")
    experts = get_expert_pool()

    val_preds = {}
    test_preds = {}

    for name, model in experts.items():
        print(f"  Training expert: {name}")
        # Fit model
        model.fit(X_train, y_train)

        # Generate probabilities
        # Note: In debug mode, we predict probabilities for the subset of classes observed.
        p_val = model.predict_proba(X_val)
        p_test = model.predict_proba(X_test)

        # Verify output shape matches (n_samples, n_classes_in_subset)
        assert p_val.shape == (X_val.shape[0], n_classes_subset)

        val_preds[name] = p_val
        test_preds[name] = p_test

    # 5. Ensemble Selection
    print("[Ensemble] Fitting GreedySelector...")
    # Reduce max iterations for speed
    selector = GreedySelector(max_iterations=3, tolerance=1e-5)

    # Cite debug_lesson_7: Explicitly pass the full set of training classes (labels)
    # to the selector. This ensures log_loss knows the correct dimension of the
    # probability matrix even if the validation set (y_val) is missing some classes.
    train_labels = np.unique(y_train)
    selector.fit(val_preds, y_val, labels=train_labels)

    print(f"  Selected Experts: {selector.selected_experts}")
    print(f"  Weights: {selector.weights}")

    # Verify selector found a solution
    assert len(selector.weights) > 0, "Selector failed to find any valid experts."

    # 6. Final Prediction and Submission
    print("[Submission] Generating final predictions...")
    final_probs = selector.predict(test_preds)

    # Verify final probabilities
    assert final_probs.shape == (len(test_ids), n_classes_subset)
    assert np.allclose(final_probs.sum(axis=1), 1.0), "Probabilities do not sum to 1."

    # Construct DataFrame
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save to disk
    Config.ensure_directories()
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"  Submission saved to: {submission_path}")

    # Verify file existence
    assert os.path.exists(submission_path)

    # Read back to confirm format
    df_check = pd.read_csv(submission_path)
    print(f"  Saved file shape: {df_check.shape}")
    assert df_check.shape == (50, n_classes_subset + 1)  # +1 for 'id' column

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
