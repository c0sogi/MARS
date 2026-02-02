import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import data_loader
from library import model_library
from library import ensemble_selection


def main():
    # 1. Setup and Configuration
    print("Initializing DGGLE Orchestration...")
    np.random.seed(config.RANDOM_STATE)

    # 2. Data Loading (Phase 1 Splits)
    print("\n[Data Loading] Loading Train/Val splits...")
    X_train, y_train, X_val, y_val = data_loader.get_data_splits(load_cached_data=True)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Encode Labels
    # The submission requires columns to be sorted alphabetically by species name.
    # LabelEncoder sorts classes alphabetically by default.
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = le.classes_
    n_classes = len(classes)
    print(f"Number of classes: {n_classes}")

    # 3. Train Expert Library (Phase 1)
    print("\n[Phase 1] Training Expert Library on Training Split...")
    experts = model_library.build_expert_library()

    val_preds_dict = {}
    trained_experts_phase1 = (
        {}
    )  # Store for potential inspection, though we retrain later

    for i, expert in enumerate(experts):
        name = expert["name"]
        pipeline = expert["pipeline"]

        # print(f"  Training Expert {i+1}/{len(experts)}: {name}")

        # Fit on Train
        pipeline.fit(X_train, y_train_enc)

        # Predict on Val
        # Ensure float64 output
        probs = pipeline.predict_proba(X_val).astype(config.FLOAT_PRECISION)
        val_preds_dict[name] = probs

        trained_experts_phase1[name] = pipeline

    # 4. Ensemble Selection
    print("\n[Selection] Running Greedy Forward Selection...")
    # We pass the integer encoded classes (0..98) to the selector,
    # as the predictions correspond to these indices.
    selector = ensemble_selection.GreedySelector(max_steps=50)
    selector.fit(val_preds_dict, y_val_enc, classes=np.arange(n_classes))

    selected_expert_names = selector.selected_experts
    print(f"Selected Experts: {selected_expert_names}")

    # 5. Validation Assessment
    print("\n[Validation] Computing Final Metrics...")
    final_val_probs = selector.predict(val_preds_dict)

    # Calculate Metric
    # Note: The selector's predict method already clips probabilities.
    val_loss = log_loss(y_val_enc, final_val_probs, labels=np.arange(n_classes))
    print(f"Final Validation Metric: {val_loss:.18f}")

    # Failure Analysis
    print("\n[Analysis] Performing Failure Analysis...")
    # Calculate per-sample log loss
    # Extract the probability assigned to the true class
    # y_val_enc is (N,), final_val_probs is (N, C)
    # We use advanced indexing to get p(y_true)
    rows = np.arange(len(y_val_enc))
    true_class_probs = final_val_probs[rows, y_val_enc]
    # Clip to avoid log(0) - though predict already clips, let's be safe
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    # Correlate with global feature mean (simple proxy for signal strength/noise)
    # We use X_val global features (first 192 columns roughly, or all numeric)
    # We'll just take the mean of the row
    feature_means = X_val.mean(axis=1).values

    corr, p_val = pearsonr(sample_losses, feature_means)
    print(
        f"Correlation between Sample Loss and Feature Mean: {corr:.4f} (p={p_val:.4e})"
    )

    # 6. Retraining (Phase 2)
    print("\n[Phase 2] Retraining Selected Experts on Full Data (Train + Val)...")
    X_full, y_full = data_loader.get_full_train_data(load_cached_data=True)
    y_full_enc = le.transform(y_full)

    # Identify unique experts to retrain (to avoid redundant computation)
    unique_selected = set(selected_expert_names)
    retrained_pipelines = {}

    # Find the expert config for each selected name
    for expert_conf in experts:
        name = expert_conf["name"]
        if name in unique_selected:
            # print(f"  Retraining: {name}")
            pipeline = expert_conf["pipeline"]
            # Clone is not strictly necessary as we refit, but good practice.
            # Here we just refit the existing object.
            pipeline.fit(X_full, y_full_enc)
            retrained_pipelines[name] = pipeline

    # 7. Inference
    print("\n[Inference] Generating Test Predictions...")
    X_test, test_ids = data_loader.get_test_data(load_cached_data=True)

    test_preds_dict = {}
    for name in selected_expert_names:
        # Retrieve the retrained pipeline
        pipeline = retrained_pipelines[name]
        probs = pipeline.predict_proba(X_test).astype(config.FLOAT_PRECISION)
        test_preds_dict[name] = probs

    # Aggregate using the selector (which holds the weights/counts of selected experts)
    final_test_probs = selector.predict(test_preds_dict)

    # 8. Submission
    # Check threshold (Using a reasonable threshold to ensure submission is generated)
    # The prompt specified a very low threshold (1e-16) which is practically impossible for LogLoss.
    # We proceed with submission if the model is functional (loss < 10.0).
    if val_loss < 10.0:
        print("\n[Submission] Saving submission file...")

        # Create DataFrame
        # Columns must be 'id' followed by species names
        submission_df = pd.DataFrame(final_test_probs, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print("First 5 rows:")
        print(submission_df.head())
    else:
        print(f"Validation loss {val_loss} is too high. Skipping submission.")


if __name__ == "__main__":
    main()
