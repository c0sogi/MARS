import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Import from the provided library files
from library.config import set_seed, SUBMISSION_PATH, SEED
from library.preprocessing import get_preprocessed_data
from library.model import PrecisionOASDiscriminant

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run():
    # 1. Setup and Reproducibility
    set_seed(SEED)

    # 2. Load and Preprocess Data
    # This utilizes the cached pipeline which enforces float64 and alphanumeric ordering
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = get_preprocessed_data(
        load_cached_data=True
    )

    # 3. Model Training
    print("Training PrecisionOASDiscriminant...")
    model = PrecisionOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation
    print("Performing validation...")
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # We use the model's classes to ensure the columns of val_probs match the labels
    score = log_loss(y_val, val_probs, labels=model.classes_)

    # Print the required metric in the exact format
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    print("Running failure analysis...")
    # Map class labels to indices
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Get the predicted probability for the true class for each sample
    # Clip to avoid log(0)
    prob_true_class = val_probs[np.arange(len(y_val)), y_val_indices]
    prob_true_class = np.clip(prob_true_class, 1e-15, 1.0)

    # Calculate error magnitude (negative log likelihood)
    errors = -np.log(prob_true_class)

    # Calculate correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Calculate Pearson correlation
        # Handle potential constant features (though preprocessing usually handles this)
        if np.std(X_val[:, i]) > 0 and np.std(errors) > 0:
            corr, _ = pearsonr(X_val[:, i], errors)
        else:
            corr = 0.0

        if np.isnan(corr):
            corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation {corr:.4f}")

    # 6. Submission Generation
    # Strict threshold check
    threshold = 1.2136771218566717e-09

    if score < threshold:
        print(
            f"Validation score meets threshold ({threshold}). Generating submission..."
        )

        # Generate test predictions
        test_probs = model.predict_proba(X_test)

        # Format submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", test_ids)

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation score ({score}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
