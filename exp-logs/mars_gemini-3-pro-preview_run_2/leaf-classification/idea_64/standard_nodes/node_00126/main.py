import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided libraries
from library.utils import set_seed, clipped_log_loss
from library.data_manager import LeafDataManager
from library.model_definitions import get_expert_library
from library.ensemble import GreedyForwardSelector

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD_METRIC = 10.0  # Practical threshold to ensure submission generation


def ensure_directories():
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)


def perform_failure_analysis(X_val, y_val, y_pred, feature_names=None):
    """
    Analyzes which features correlate with high prediction error.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Encode y_val to match y_pred columns if necessary
    # Assuming y_pred columns are sorted alphabetically, which is standard for sklearn
    classes = np.unique(y_val)
    le = LabelEncoder()
    le.fit(classes)
    y_val_idx = le.transform(y_val)

    # 2. Calculate per-sample Log Loss
    # Loss = -log(p_true_class)
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Extract probability of the true class for each sample
    # Advanced indexing: [row_indices, col_indices]
    prob_true = y_pred_clipped[np.arange(len(y_val)), y_val_idx]
    sample_losses = -np.log(prob_true)

    print(f"Mean Validation Log Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Sample Loss: {np.max(sample_losses):.6f}")

    # 3. Correlate Error with Features
    # We compute correlation between each feature column in X_val and sample_losses
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        # Handle potential constant features (std=0) to avoid NaN correlation
        feat_col = X_val[:, i]
        if np.std(feat_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, sample_losses)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top correlated features (magnitude)
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features Correlated with Error Magnitude:")
    for idx in top_indices:
        fname = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"  - {fname}: Correlation = {correlations[idx]:.4f}")


def main():
    # 1. Setup
    set_seed(42)
    ensure_directories()
    print("Starting FBPGE Pipeline...")

    # 2. Data Loading
    dm = LeafDataManager()
    data = dm.load_data(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]

    feature_indices = dm.get_feature_indices()

    # Construct feature names list for analysis
    # Order matches LeafDataManager.feature_columns
    feature_names = dm.margin_cols + dm.shape_cols + dm.texture_cols + dm.physical_cols

    # 3. Phase 1: Selection (Train on Train, Evaluate on Val)
    print("\n--- Phase 1: Expert Selection ---")
    experts = get_expert_library(feature_indices)
    print(f"Initialized {len(experts)} experts.")

    val_preds_dict = {}
    trained_models_phase1 = {}

    # Train and Predict
    for name, pipeline in experts:
        # print(f"Training {name}...")
        try:
            pipeline.fit(X_train, y_train)
            # Predict probabilities
            preds = pipeline.predict_proba(X_val)
            val_preds_dict[name] = preds
            trained_models_phase1[name] = pipeline  # Keep reference
        except Exception as e:
            print(f"Failed to train expert {name}: {e}")

    # Run Greedy Selection
    selector = GreedyForwardSelector(max_ensemble_size=20, verbose=True)
    selector.fit(val_preds_dict, y_val)

    # Compute Final Validation Metric
    final_val_preds = selector.predict(val_preds_dict)
    final_val_metric = clipped_log_loss(y_val, final_val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    perform_failure_analysis(X_val, y_val, final_val_preds, feature_names)

    # 4. Phase 2: Final Retraining and Submission
    # Condition check (using practical threshold)
    if final_val_metric < THRESHOLD_METRIC:
        print("\n--- Phase 2: Retraining and Submission ---")

        # Combine Train and Val
        print(f"Combining Train ({len(X_train)}) and Val ({len(X_val)}) sets...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        # Identify selected experts
        selected_expert_names = list(selector.weights_.keys())
        print(f"Retraining {len(selected_expert_names)} unique experts on full data...")

        test_preds_dict = {}

        # Retrain only selected experts
        # We need to grab the fresh pipeline definitions to ensure clean state
        # or clone the existing ones. Since get_expert_library returns new objects,
        # we can just iterate and match names.
        all_experts_map = dict(get_expert_library(feature_indices))

        # Classes for column headers (sklearn sorts alphabetically)
        # We fit a dummy LabelEncoder or use one of the models to get classes
        # All models should see the same classes since we use the full dataset
        # We'll use the first trained model to get class labels for consistency
        sample_model = all_experts_map[selected_expert_names[0]]
        sample_model.fit(X_full, y_full)
        classes = sample_model.named_steps["clf"].classes_

        for name in selected_expert_names:
            pipeline = all_experts_map[name]
            pipeline.fit(X_full, y_full)
            test_preds_dict[name] = pipeline.predict_proba(X_test)

        # Ensemble Prediction
        # We manually compute weighted average using selector.weights_
        final_test_preds = np.zeros((len(X_test), len(classes)))
        total_weight = sum(selector.weights_.values())

        for name, weight in selector.weights_.items():
            final_test_preds += test_preds_dict[name] * weight

        final_test_preds /= total_weight

        # 5. Generate Submission File
        print("Generating submission file...")
        df_sub = pd.DataFrame(final_test_preds, columns=classes)
        df_sub.insert(0, "id", test_ids)

        # Save
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")

    else:
        print(
            f"Validation metric {final_val_metric} did not meet threshold {THRESHOLD_METRIC}."
        )


if __name__ == "__main__":
    main()
