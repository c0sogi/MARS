import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.preprocessing as preprocessing
import library.model as model_lib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # 1. Initialization
    set_seed(config.SEED)
    print("Starting execution pipeline...")

    # 2. Data Loading
    # Use load_cached_data=True to utilize pre-computed parquet/npy files
    print("[1/6] Loading data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, classes = (
        data_loader.load_data(load_cached_data=True)
    )

    # 3. Preprocessing
    # Apply Yeo-Johnson and Standard Scaling
    print("[2/6] Preprocessing data...")
    X_train, X_val, X_test = preprocessing.preprocess_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    # Train the Cholesky-Solved Exact-Precision OAS Discriminant
    print("[3/6] Training model...")
    model = model_lib.CholeskyOASDiscriminant()
    model.fit(X_train, y_train)

    # 5. Validation
    print("[4/6] Validating...")
    val_probs = model.predict_proba(X_val)

    # Transform validation labels to integer indices matching the model's classes
    # The model stores its internal LabelEncoder in model.le_
    y_val_indices = model.le_.transform(y_val)

    # Calculate Log Loss with full float64 precision
    val_loss = log_loss(y_val_indices, val_probs, labels=range(len(model.classes_)))

    # Print the required metric string with high precision
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("[5/6] Performing failure analysis...")
    # Calculate error magnitude: 1.0 - probability assigned to the true class
    # val_probs shape: (N_samples, N_classes)
    true_class_probs = val_probs[np.arange(len(y_val_indices)), y_val_indices]
    error_magnitudes = 1.0 - true_class_probs

    # Calculate correlation between features and error magnitude
    feature_names = X_val_raw.columns.tolist()
    correlations = []

    for i in range(X_val.shape[1]):
        feature_col = X_val[:, i]
        # Avoid correlation calculation on constant features
        if np.std(feature_col) > 1e-15:
            corr, _ = pearsonr(feature_col, error_magnitudes)
            correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # 7. Submission
    print("[6/6] Checking submission criteria...")
    # Strict threshold from task description
    THRESHOLD = 1.2136771218566717e-09

    if val_loss < THRESHOLD:
        print(
            f"Validation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for test set
        test_probs = model.predict_proba(X_test)

        # Construct submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", test_ids)

        # Save to CSV
        submission_df.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to: {config.SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
