import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Import from library
from library.config import RANDOM_SEED, SUBMISSION_PATH, FLOAT_PRECISION
from library.data import load_data
from library.experts import build_expert_library
from library.ensemble import GreedySelector

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)


def main():
    print("Starting Runfile...")

    # 1. Load Data
    # -------------------------------------------------------------------------
    # Load cached data (Global features + Extracted Physical features)
    data = load_data(load_cached_data=True)

    X_train_global = data["X_train_global"]
    X_train_physical = data["X_train_physical"]
    y_train = data["y_train"]

    X_val_global = data["X_val_global"]
    X_val_physical = data["X_val_physical"]
    y_val = data["y_val"]

    X_test_global = data["X_test_global"]
    X_test_physical = data["X_test_physical"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    # 2. Phase 1: Ensemble Selection (Internal Split)
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Ensemble Selection ===")

    # Split the provided training data into internal train/selection sets
    # This ensures we select experts based on unseen data relative to the internal training
    indices = np.arange(len(y_train))
    train_idx, sel_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # Create internal subsets
    X_sub_train_global = X_train_global[train_idx]
    X_sub_train_physical = X_train_physical[train_idx]
    y_sub_train = y_train[train_idx]

    X_sub_sel_global = X_train_global[sel_idx]
    X_sub_sel_physical = X_train_physical[sel_idx]
    y_sub_sel = y_train[sel_idx]

    # Build the library of candidate experts
    experts = build_expert_library()
    print(f"Initialized {len(experts)} experts.")

    # Train all experts on sub-train and predict on sub-selection
    expert_preds_sel = {}

    for expert in experts:
        # Select appropriate feature view based on expert type
        if expert.feature_type == "global":
            X_fit = X_sub_train_global
            X_pred = X_sub_sel_global
        elif expert.feature_type == "physical":
            X_fit = X_sub_train_physical
            X_pred = X_sub_sel_physical
        else:
            continue

        # Fit expert
        expert.fit(X_fit, y_sub_train)

        # Generate predictions
        preds = expert.predict_proba(X_pred)
        expert_preds_sel[expert.name] = preds

    # Run Greedy Forward Selection to find optimal ensemble
    selector = GreedySelector()
    selector.fit(expert_preds_sel, y_sub_sel)

    selected_names = selector.selected_experts
    print(f"Selected experts: {selected_names}")

    # Identify unique selected experts for efficient retraining
    unique_selected_names = set(selected_names)
    selected_experts_objs = [e for e in experts if e.name in unique_selected_names]

    # 3. Phase 2: Validation (Full Train -> Metadata Val)
    # -------------------------------------------------------------------------
    print("\n=== Phase 2: Validation ===")

    val_preds_dict = {}

    # Retrain selected experts on the full training set (metadata train)
    for expert in selected_experts_objs:
        if expert.feature_type == "global":
            X_fit = X_train_global
            X_pred = X_val_global
        elif expert.feature_type == "physical":
            X_fit = X_train_physical
            X_pred = X_val_physical
        else:
            continue

        expert.fit(X_fit, y_train)
        preds = expert.predict_proba(X_pred)
        val_preds_dict[expert.name] = preds

    # Aggregate predictions using the selector (handles weighting)
    final_val_probs = selector.predict(val_preds_dict)

    # Compute and print final metric
    val_loss = log_loss(y_val, final_val_probs, labels=range(len(classes)))
    print(f"Final Validation Metric: {val_loss:.20f}")

    # 4. Phase 3: Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Failure Analysis ===")

    # Calculate per-sample log loss for correlation analysis
    rows = np.arange(len(y_val))
    # Extract probability assigned to the true class
    true_class_probs = final_val_probs[rows, y_val]
    # Clip for stability
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    # Compute correlation with Global Features
    n_features = X_val_global.shape[1]
    correlations = []

    for i in range(n_features):
        feat_vals = X_val_global[:, i]
        # Avoid division by zero for constant features
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, sample_losses)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Report Top 5 Positive Correlations (High Feature Value -> High Error)
    top_pos_idx = np.argsort(correlations)[-5:][::-1]
    print("Top 5 Features associated with High Error (Positive Corr):")
    for idx in top_pos_idx:
        print(f"  Feature {idx}: {correlations[idx]:.4f}")

    # Report Top 5 Negative Correlations (Low Feature Value -> High Error)
    top_neg_idx = np.argsort(correlations)[:5]
    print("Top 5 Features associated with High Error (Negative Corr):")
    for idx in top_neg_idx:
        print(f"  Feature {idx}: {correlations[idx]:.4f}")

    # 5. Phase 4: Submission (Train+Val -> Test)
    # -------------------------------------------------------------------------
    print("\n=== Phase 4: Submission ===")

    # Combine Training and Validation sets for final model training
    X_combined_global = np.vstack([X_train_global, X_val_global])
    X_combined_physical = np.vstack([X_train_physical, X_val_physical])
    y_combined = np.hstack([y_train, y_val])

    test_preds_dict = {}

    # Retrain selected experts on combined data
    for expert in selected_experts_objs:
        if expert.feature_type == "global":
            X_fit = X_combined_global
            X_pred = X_test_global
        elif expert.feature_type == "physical":
            X_fit = X_combined_physical
            X_pred = X_test_physical
        else:
            continue

        expert.fit(X_fit, y_combined)
        preds = expert.predict_proba(X_pred)
        test_preds_dict[expert.name] = preds

    # Aggregate predictions
    final_test_probs = selector.predict(test_preds_dict)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
