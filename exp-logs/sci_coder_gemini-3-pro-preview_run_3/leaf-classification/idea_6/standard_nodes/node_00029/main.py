import sys
import numpy as np
import warnings
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders
from library.feature_extraction import process_split
from library.ensemble import BaggedLDAPipeline


def main():
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # 1. Setup & Reproducibility
    seed_everything()

    # 2. Data Loading
    # Loaders handle reading CSVs and image processing configuration
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # 3. Feature Extraction
    # We use load_cached_data=True to utilize features computed in previous steps/ideas
    # if they exist in the ./working/idea_6 (or configured cache) directory.

    # Train Split
    X_train_dino, X_train_conv, X_train_tab, y_train, _ = process_split(
        train_loader, "train", load_cached_data=True
    )

    # Validation Split
    X_val_dino, X_val_conv, X_val_tab, y_val, val_ids = process_split(
        val_loader, "val", load_cached_data=True
    )

    # Test Split (needed for submission)
    X_test_dino, X_test_conv, X_test_tab, _, test_ids = process_split(
        test_loader, "test", load_cached_data=True
    )

    # 4. Model Training
    # Initialize the Homogeneous Bagged Ensemble of LDA
    model = BaggedLDAPipeline()
    model.fit(X_train_dino, X_train_conv, X_train_tab, y_train)

    # 5. Validation Inference & Metric
    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val_dino, X_val_conv, X_val_tab)

    # Calculate Multi-class Log Loss
    # We provide labels=range(len(classes)) to ensure correct handling if a class is missing in val
    metric = log_loss(y_val, val_probs, labels=range(len(classes)))

    # Print the required metric string
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    # Calculate per-sample loss: -log(p_true)
    # Clip probabilities to avoid log(0)
    epsilon = Config.PROB_EPSILON
    val_probs_clipped = np.clip(val_probs, epsilon, 1.0 - epsilon)

    # Extract probability assigned to the true class for each sample
    # y_val contains class indices
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    print("\nFailure Analysis (Correlation with Tabular Features):")

    # Construct feature names for the tabular data (192 features)
    feature_names = (
        [f"margin_{i+1}" for i in range(64)]
        + [f"shape_{i+1}" for i in range(64)]
        + [f"texture_{i+1}" for i in range(64)]
    )

    correlations = []
    # Compute correlation between each feature and the loss
    for i in range(X_val_tab.shape[1]):
        feat_vals = X_val_tab[:, i]
        # Skip constant features to avoid warnings
        if np.std(feat_vals) < 1e-12:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_vals, sample_losses)
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation magnitude (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 10 correlations
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    # Generate predictions for test set
    test_probs = model.predict_proba(X_test_dino, X_test_conv, X_test_tab)

    # Save submission file
    save_submission(test_probs, test_ids, classes)


if __name__ == "__main__":
    main()
