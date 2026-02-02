import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided libraries
import library.config as conf
import library.engine as engine
import library.model_factory as model_factory

# Set seeds for reproducibility
np.random.seed(conf.RANDOM_SEED)


def run():
    # Initialize PhaseManager
    pm = engine.PhaseManager()

    # 1. Load Data
    print("Loading data...")
    # Using load_cached_data=True as requested to speed up execution
    (
        X_train_views,
        y_train,
        X_val_views,
        y_val,
        X_test_views,
        test_ids,
        classes,
    ) = pm.data_manager.load_all_data(load_cached_data=True)

    # 2. Phase A: Expert Selection
    print("\n" + "=" * 40)
    print("PHASE A: Expert Selection")
    print("=" * 40)

    # Get expert pool
    pool = model_factory.get_expert_pool()

    # Train candidates on Training set
    # The dataset is small (~700 samples), so we use the full training set
    # to ensure the best possible model performance.
    print(f"Training {len(pool)} candidate experts...")
    fitted_models_a = pm.train_pool(pool, X_train_views, y_train)

    # Generate predictions on Validation set
    print("Generating validation predictions...")
    val_preds_dict = pm.generate_predictions(fitted_models_a, X_val_views)

    # Run Greedy Forward Selection
    print("Running Greedy Ensemble Selection...")
    pm.selector.fit(val_preds_dict, y_val)

    # Retrieve Validation Metric
    # The selector stores the best score (log loss) found during the greedy process
    final_val_metric = pm.selector.best_score_

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_val_metric}")

    # 3. Failure Analysis
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Get ensemble probability predictions for validation set
    val_probs = pm.selector.predict(val_preds_dict)

    # Calculate per-sample log loss (Error Magnitude)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Create one-hot encoding of true labels
    n_samples = len(y_val)
    n_classes = len(classes)
    y_val_ohe = np.zeros((n_samples, n_classes))
    y_val_ohe[np.arange(n_samples), y_val] = 1

    # Compute cross-entropy loss per sample
    sample_losses = -np.sum(y_val_ohe * np.log(val_probs_clipped), axis=1)

    # Correlate error with features (Global View contains all features)
    X_val_global = X_val_views["Global"]
    feature_names = conf.ALL_FEATURE_COLS

    correlations = []
    # Calculate correlation for each feature
    for i in range(X_val_global.shape[1]):
        feat_values = X_val_global[:, i]
        # Handle constant features to avoid warning
        if np.std(feat_values) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(sample_losses, feat_values)
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 4. Conditional Submission
    THRESHOLD = 3.3768230269455483e-06

    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Proceeding to Phase B: Retraining and Submission...")

        print("\n" + "=" * 40)
        print("PHASE B: Final Retraining")
        print("=" * 40)

        # Combine Train and Val data
        X_full_views = {}
        for view_name in X_train_views.keys():
            X_full_views[view_name] = np.vstack(
                [X_train_views[view_name], X_val_views[view_name]]
            )
        y_full = np.concatenate([y_train, y_val])

        # Identify selected models
        selected_model_names = pm.selector.selected_models_
        unique_selected = sorted(list(set(selected_model_names)))
        print(f"Retraining {len(unique_selected)} unique selected models...")

        # Retrain selected models
        # This method handles hyperparameter transfer (C) for LR models
        final_models = pm.retrain_selected(
            unique_selected, fitted_models_a, X_full_views, y_full
        )

        # Inference on Test Set
        print("\n" + "=" * 40)
        print("INFERENCE & SUBMISSION")
        print("=" * 40)
        print("Generating test predictions...")
        test_preds_dict = pm.generate_predictions(final_models, X_test_views)

        # Aggregate using the ensemble weights
        final_probs = pm.selector.predict(test_preds_dict)

        # Save Submission
        pm.save_submission(test_ids, classes, final_probs)

    else:
        print(
            f"\nValidation metric ({final_val_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
