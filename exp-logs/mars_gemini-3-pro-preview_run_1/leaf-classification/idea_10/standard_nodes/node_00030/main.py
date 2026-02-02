import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library import config
from library import dataset
from library import transforms
from library import classifier


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_correlation(x, y):
    """Calculates Pearson correlation between two 1D numpy arrays."""
    if len(x) != len(y):
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    # Set reproducible state
    set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    # dataset.load_dataset handles geometric augmentation and caching
    train_data, val_data, test_data = dataset.load_dataset(load_cached_data=True)

    X_train_raw, y_train, train_ids = train_data
    X_val_raw, y_val, val_ids = val_data
    X_test_raw, test_ids = test_data

    # Store feature names for failure analysis
    feature_names = X_train_raw.columns.tolist()

    # -------------------------------------------------------------------------
    # 2. Preprocessing (Power Transform + Scaling)
    # -------------------------------------------------------------------------
    # Check for cached transformed features to speed up execution
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_X_train = os.path.join(config.WORKING_DIR, "X_train_transformed.npy")
    cache_X_val = os.path.join(config.WORKING_DIR, "X_val_transformed.npy")
    cache_X_test = os.path.join(config.WORKING_DIR, "X_test_transformed.npy")

    if (
        os.path.exists(cache_X_train)
        and os.path.exists(cache_X_val)
        and os.path.exists(cache_X_test)
    ):
        X_train = np.load(cache_X_train)
        X_val = np.load(cache_X_val)
        X_test = np.load(cache_X_test)
    else:
        # Fit pipeline on training data and transform all splits
        pipeline = transforms.get_pipeline()
        pipeline.fit(X_train_raw)

        X_train = pipeline.transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)
        X_test = pipeline.transform(X_test_raw)

        # Cache results
        np.save(cache_X_train, X_train)
        np.save(cache_X_val, X_val)
        np.save(cache_X_test, X_test)

    # -------------------------------------------------------------------------
    # 3. Model Training (Standard LDA)
    # -------------------------------------------------------------------------
    print(
        f"Initializing LeafLDA with solver={config.LDA_SOLVER}, shrinkage={config.LDA_SHRINKAGE}"
    )
    model = classifier.LeafLDA()

    # Cite solution_lesson_node_00028: Avoid Transductive Learning in this regime.
    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 4. Validation
    # -------------------------------------------------------------------------
    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)

    # Clip probabilities to avoid log(0) and match metric definition
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Calculate Multi-class Log Loss
    loss = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {loss}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    # Map class labels to indices
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = y_val.map(class_to_idx).values

    # Get probability assigned to the true class for each sample
    prob_true = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Calculate error magnitude (Negative Log Likelihood)
    errors = -np.log(prob_true)

    # Calculate correlation between features and error
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Skip constant features
        if np.std(feature_vals) < 1e-9:
            corr = 0.0
        else:
            corr = calculate_correlation(feature_vals, errors)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with error:")
    for idx, corr in correlations[:5]:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {feat_name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    threshold = 1.470544781593644e-08

    if loss < threshold:
        # Generate predictions for test set
        test_probs = model.predict_proba(X_test)

        # Load sample submission to ensure correct format
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

        # Create DataFrame with model predictions
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", test_ids.values)

        # Identify required target columns
        target_cols = [c for c in sample_sub.columns if c != "id"]

        # Ensure all columns exist (fill 0 for classes not present in training)
        missing_cols = set(target_cols) - set(submission_df.columns)
        for c in missing_cols:
            submission_df[c] = 0.0

        # Reorder columns to match sample submission
        submission_df = submission_df[["id"] + target_cols]

        # Clip probabilities
        submission_df[target_cols] = submission_df[target_cols].clip(
            lower=epsilon, upper=1 - epsilon
        )

        # Save submission
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
    else:
        print(
            f"Validation metric {loss} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
