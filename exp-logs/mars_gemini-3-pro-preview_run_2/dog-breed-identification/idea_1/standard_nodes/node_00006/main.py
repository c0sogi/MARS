import numpy as np
import sys
from library.config import SUBMISSION_PATH, SEED
from library.utils import set_seed, save_submission
from library.data_loader import create_dataloaders
from library.feature_extractor import FeatureExtractor
from library.classifier import LogRegClassifier


def main():
    # 1. Setup
    set_seed(SEED)
    print("Starting execution...")

    # 2. Data Loading
    # We use the full dataset because the feature-extraction + logistic regression
    # workflow is extremely efficient (minutes) compared to end-to-end CNN training.
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        debug_limit=None
    )

    # 3. Feature Extraction
    # This step handles GPU acceleration and caching automatically.
    # First run will compute features; subsequent runs will load from ./working
    extractor = FeatureExtractor()

    print("Extracting/Loading training features...")
    X_train, y_train = extractor.get_train_features(
        train_loader, load_cached_data=False
    )

    print("Extracting/Loading validation features...")
    X_val, y_val = extractor.get_val_features(val_loader, load_cached_data=False)

    print("Extracting/Loading test features...")
    X_test, test_ids = extractor.get_test_features(test_loader, load_cached_data=False)

    # 4. Training
    classifier = LogRegClassifier()
    # Train the logistic regression head on the fixed ResNet features
    classifier.train(X_train, y_train, load_cached_model=False)

    # 5. Validation and Failure Analysis
    # Compute overall metric
    val_loss = classifier.evaluate(X_val, y_val)
    print(f"Final Validation Metric: {val_loss}")

    # Failure Analysis: Correlate error with input feature properties
    print("\nPerforming Failure Analysis...")

    # Predict probabilities on validation set
    val_probs = classifier.predict(X_val)

    # Calculate per-sample Log Loss
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Extract the probability assigned to the true class for each sample
    # y_val contains the true class indices
    sample_indices = np.arange(len(y_val))
    true_class_probs = val_probs_clipped[sample_indices, y_val.astype(int)]

    # Loss = -log(p_true)
    per_sample_loss = -np.log(true_class_probs)

    # Calculate L2 norm of the feature vectors (signal magnitude)
    feature_norms = np.linalg.norm(X_val, axis=1)

    # Calculate correlation
    if len(per_sample_loss) > 1:
        correlation = np.corrcoef(per_sample_loss, feature_norms)[0, 1]
        print(
            f"Correlation between Error Magnitude (Log Loss) and Feature Vector Norm: {correlation:.6f}"
        )

        # Also check correlation with maximum activation value
        feature_max = np.max(X_val, axis=1)
        corr_max = np.corrcoef(per_sample_loss, feature_max)[0, 1]
        print(
            f"Correlation between Error Magnitude and Max Feature Activation: {corr_max:.6f}"
        )
    else:
        print("Not enough validation samples for correlation analysis.")

    # 6. Submission
    baseline_metric = 0.15753624086163448
    if val_loss < baseline_metric:
        print("\nGenerating submission...")
        test_probs = classifier.predict(X_test)

        # The classifier outputs probabilities for classes in index order (0..N)
        # The 'classes' list from data_loader maps index -> breed name
        save_submission(test_probs, test_ids, classes, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {val_loss} is not lower than baseline {baseline_metric}. Skipping submission."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
