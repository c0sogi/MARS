import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library import config
from library import data_processing
from library import training
from library import model as model_lib


def run():
    # Ensure reproducibility
    np.random.seed(config.SEED)

    # 1. Load Data
    # Utilizing cached data for speed as per instructions
    print("Loading data...")
    data = data_processing.process_data(load_cached_data=True)
    (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        ids_test,
        classes,
    ) = data

    # 2. Hyperparameter Optimization
    # Find the best shrinkage parameter lambda using Stratified K-Fold CV
    print("Optimizing shrinkage parameter...")
    best_lambda = training.optimize_shrinkage(
        X_train,
        y_train,
        genus_train,
        classes,
        lambda_values=np.linspace(0, 0.5, 11),
        n_splits=5,
        seed=config.SEED,
    )

    # 3. Final Model Training
    # Train the model on the training set using the optimal lambda
    print(f"Training final model with lambda={best_lambda:.4f}...")
    clf = model_lib.TaxonomicDualCentroidOAS(lambda_reg=best_lambda)
    clf.fit(X_train, y_train, genus_train)

    # 4. Validation Evaluation
    # Predict probabilities on the hold-out validation set
    print("Evaluating on validation set...")
    val_probs = clf.predict_proba(X_val)

    # Compute Multi-class Log Loss
    # Scikit-learn's log_loss handles the clipping/epsilon internally (eps=1e-15 default)
    val_metric = log_loss(y_val, val_probs, labels=classes)

    # Print the required metric string
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss to correlate with features
    # Map string labels to integer indices matching the 'classes' array order
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract predicted probability for the true class for each sample
    # val_probs shape: (n_samples, n_classes)
    true_class_probs = val_probs[np.arange(len(y_val)), y_val_indices]

    # Clip probabilities for numerical stability in log calculation
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)

    # Compute Cross-Entropy Loss per sample
    sample_losses = -np.log(true_class_probs)

    # Compute correlation between each feature and the error vector
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Check for constant features to avoid division by zero in correlation
        if np.std(X_val[:, i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X_val[:, i], sample_losses)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify features most strongly associated with error
    # Positive correlation: High feature value -> High Error
    top_pos_indices = np.argsort(correlations)[-5:][::-1]
    # Negative correlation: Low feature value -> High Error
    top_neg_indices = np.argsort(correlations)[:5]

    print("Top features positively correlated with error (High Value -> High Error):")
    for idx in top_pos_indices:
        print(f"  {config.FEATURES[idx]}: {correlations[idx]:.4f}")

    print("Top features negatively correlated with error (Low Value -> High Error):")
    for idx in top_neg_indices:
        print(f"  {config.FEATURES[idx]}: {correlations[idx]:.4f}")

    # 6. Submission Generation
    # Strict threshold check as per task description
    THRESHOLD = 1.2136771218566717e-09

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        training.generate_submission(
            clf, X_test, ids_test, classes, config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
