import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import model_factory
from library import data_manager


def set_seeds(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clip_probabilities(probas):
    """
    Clips probabilities to [1e-15, 1 - 1e-15] as per task description.
    """
    eps = 1e-15
    return np.clip(probas, eps, 1 - eps)


def prepare_validation_data():
    """
    Loads train and val metadata separately for validation assessment.
    Mirrors the logic in data_manager.py but keeps splits distinct.
    """
    # Load metadata
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(config.VAL_META_PATH)

    # Identify feature columns
    non_feature_cols = ["id", "species", "image_path"]
    feature_cols = [c for c in df_train_meta.columns if c not in non_feature_cols]
    feature_cols = sorted(feature_cols)  # Critical: Ensure sorted order

    # Prepare arrays
    X_train = df_train_meta[feature_cols].values.astype(np.float32)
    y_train_raw = df_train_meta["species"].values

    X_val = df_val_meta[feature_cols].values.astype(np.float32)
    y_val_raw = df_val_meta["species"].values

    # Encode targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    # Transform validation labels.
    # Note: Stratified split ensures coverage, but we handle potential unseen labels safely
    try:
        y_val = le.transform(y_val_raw)
    except ValueError as e:
        # Fallback: Fit on combined if train is missing classes (unlikely)
        le.fit(np.concatenate([y_train_raw, y_val_raw]))
        y_train = le.transform(y_train_raw)
        y_val = le.transform(y_val_raw)

    return X_train, y_train, X_val, y_val, feature_cols, le


def run_failure_analysis(model, X_val, y_val, feature_cols):
    """
    Analyzes correlation between error magnitude and input features.
    """
    print("\nRunning Failure Analysis...")

    # Get predictions
    probas = model.predict_proba(X_val)
    probas = clip_probabilities(probas)

    # Calculate error magnitude (1 - probability of true class)
    # Advanced indexing to get prob of true class for each sample
    true_class_probas = probas[np.arange(len(y_val)), y_val]
    errors = 1.0 - true_class_probas

    # Calculate correlation with features
    correlations = []
    for i, col_name in enumerate(feature_cols):
        feature_values = X_val[:, i]
        # Handle constant features to avoid warnings
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feature_values)

        if np.isnan(corr):
            corr = 0.0
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")


def main():
    set_seeds(config.RANDOM_STATE)

    # Threshold for submission
    SUBMISSION_THRESHOLD = 0.00870833951594525

    # 1. Prepare Data for Validation
    print("Preparing data for validation...")
    X_train_sub, y_train_sub, X_val, y_val, feature_cols, le = prepare_validation_data()

    # 2. Train Model on Subset
    print("Training SoftVotingEnsemble on training subset...")
    model = model_factory.SoftVotingEnsemble()
    model.fit(X_train_sub, y_train_sub)

    # 3. Validate
    print("Validating...")
    val_probas = model.predict_proba(X_val)
    val_probas_clipped = clip_probabilities(val_probas)

    # Calculate Log Loss
    val_loss = log_loss(y_val, val_probas_clipped, labels=np.arange(len(le.classes_)))

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_loss}")

    # 4. Failure Analysis
    run_failure_analysis(model, X_val, y_val, feature_cols)

    # 5. Submission Logic
    if val_loss < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({SUBMISSION_THRESHOLD}). Proceeding to submission..."
        )

        # Load full data (Train + Val combined)
        print("Loading full dataset for final training...")
        X_full, y_full, X_test, test_ids, classes = data_manager.load_and_prepare_data(
            load_cached_data=True
        )

        # Retrain on full data
        print("Retraining ensemble on full dataset...")
        final_model = model_factory.SoftVotingEnsemble()
        final_model.fit(X_full, y_full)

        # Predict on Test
        print("Generating test predictions...")
        test_probas = final_model.predict_proba(X_test)
        test_probas = clip_probabilities(test_probas)

        # Format Submission
        submission_df = pd.DataFrame(test_probas, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission completed successfully.")

    else:
        print(
            f"\nValidation metric ({val_loss}) did not meet threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
