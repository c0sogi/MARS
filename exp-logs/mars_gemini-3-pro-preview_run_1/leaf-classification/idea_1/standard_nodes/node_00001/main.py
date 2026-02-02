import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import load_data
from library.model import LeafModel


def main():
    # 1. Setup
    seed_everything(Config.RANDOM_SEED)

    # 2. Load Data
    # We use load_cached_data=True to leverage the preprocessed files in ./working
    print("Loading data...")
    X, y, X_test, test_ids, label_encoder = load_data(load_cached_data=True)

    # 3. Stratified K-Fold Cross-Validation
    # Using Stratified K-Fold ensures each fold has a representative distribution of the 99 classes
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    # Initialize arrays to store predictions
    num_classes = len(label_encoder.classes_)
    oof_preds = np.zeros((len(X), num_classes))
    test_preds_accumulator = np.zeros((len(X_test), num_classes))

    print(f"Starting Training with {Config.N_FOLDS} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} / {Config.N_FOLDS} ---")

        # Split data for this fold
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Initialize and Train Model
        # The LeafModel wrapper handles LightGBM initialization and params
        model = LeafModel()
        model.train(X_train, y_train, X_val, y_val)

        # Inference on Validation Fold (Out-Of-Fold predictions)
        val_probs = model.predict(X_val)
        oof_preds[val_idx] = val_probs

        # Inference on Test Set
        # We accumulate probabilities to average them later
        test_probs = model.predict(X_test)
        test_preds_accumulator += test_probs

    # 4. Aggregate Test Predictions
    avg_test_preds = test_preds_accumulator / Config.N_FOLDS

    # 5. Validation Metric Calculation
    # Apply clipping to avoid log(0) extremes, matching the competition metric definition
    epsilon = Config.PROB_CLIP_EPSILON
    oof_preds_clipped = np.clip(oof_preds, epsilon, 1.0 - epsilon)

    # Calculate Multi-class Log Loss on the entire dataset (OOF predictions)
    final_metric = log_loss(y, oof_preds_clipped, labels=np.arange(num_classes))

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude: 1.0 - probability assigned to the true class
    # We use advanced indexing to extract the predicted probability for the correct label
    true_class_probs = oof_preds_clipped[np.arange(len(y)), y]
    error_magnitude = 1.0 - true_class_probs

    # Compute correlation between input features and the error magnitude
    # X is a DataFrame, so we can use corrwith against the error Series
    error_series = pd.Series(error_magnitude, index=X.index)
    correlations = X.corrwith(error_series)

    # Sort by absolute correlation to find features most strongly related to error (positive or negative)
    abs_correlations = correlations.abs().sort_values(ascending=False)

    print("Top 5 features correlated with prediction error:")
    print(abs_correlations.head(5))

    # 7. Generate Submission
    print("\nGenerating Submission...")
    save_submission(avg_test_preds, test_ids, label_encoder)


if __name__ == "__main__":
    main()
