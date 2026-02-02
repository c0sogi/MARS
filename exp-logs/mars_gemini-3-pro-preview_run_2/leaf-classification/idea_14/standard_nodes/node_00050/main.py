import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library.config import Config
from library.data_pipeline import DataPipeline
from library.modeling import (
    get_discriminative_solver,
    get_generative_solver,
    train_predict,
)


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clip_probabilities(probas):
    """
    Clips probabilities to avoid log(0) and ensures range [0, 1].
    """
    return np.clip(probas, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)


def perform_failure_analysis(X_val, y_val, y_pred_proba, feature_names=None):
    """
    Analyzes which features correlate with higher prediction error.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate Log Loss per sample
    # Select the probability assigned to the true class
    # y_val is integer encoded, y_pred_proba is (N, n_classes)
    rows = np.arange(len(y_val))
    true_class_probs = y_pred_proba[rows, y_val]

    # Clip to avoid log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Validation Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Sample Loss: {np.max(sample_losses):.6f}")

    # 2. Correlate error with features (View 1 features are most interpretable)
    # We assume X_val is the handcrafted feature set (View 1)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_val.shape[1])]

    correlations = []
    for i in range(X_val.shape[1]):
        # Handle potential constant features or NaNs
        if np.std(X_val[:, i]) == 0:
            corr = 0
        else:
            corr = np.corrcoef(X_val[:, i], sample_losses)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")


def main():
    set_seed(Config.RANDOM_SEED)

    # Initialize Pipeline
    pipeline = DataPipeline()

    # =========================================================================
    # PHASE 1: VALIDATION (Train on 'train', Evaluate on 'val')
    # =========================================================================
    print("Starting Phase 1: Validation Assessment...")

    # 1. Load Metadata manually for strict split
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")

    df_train_split = pd.read_csv(train_meta_path)
    df_val_split = pd.read_csv(val_meta_path)

    # 2. Prepare Targets
    le = LabelEncoder()
    # Fit on combined species to ensure all classes are covered
    all_species = pd.concat(
        [df_train_split["species"], df_val_split["species"]]
    ).unique()
    le.fit(all_species)

    y_train_split = le.transform(df_train_split["species"])
    y_val_split = le.transform(df_val_split["species"])

    # 3. Process View 1: Handcrafted Features
    print("Processing View 1 (Handcrafted) for Validation...")
    X_train_v1_raw = pipeline.get_handcrafted_features(df_train_split)
    X_val_v1_raw = pipeline.get_handcrafted_features(df_val_split)

    # Scale (Fit on Train, Transform Val)
    X_train_v1, X_val_v1 = pipeline.process_view1_handcrafted(
        X_train_v1_raw, X_val_v1_raw
    )

    # 5. Train Models (Validation Split)
    print("Training Validation Models...")

    # Estimator A: LogReg on View 1
    model_a_val = get_discriminative_solver()
    probs_a_val = train_predict(
        model_a_val,
        X_train_v1,
        y_train_split,
        X_val_v1,
        model_name="Val_Estimator_A_HC_LogReg",
    )

    # Estimator B: LDA on View 1
    model_b_val = get_generative_solver()
    probs_b_val = train_predict(
        model_b_val,
        X_train_v1,
        y_train_split,
        X_val_v1,
        model_name="Val_Estimator_B_HC_LDA",
    )

    # 6. Ensemble and Evaluate
    # Soft Voting (Cite solution_lesson_node_00034)
    probs_ensemble_val = (probs_a_val + probs_b_val) / 2.0
    probs_ensemble_val = clip_probabilities(probs_ensemble_val)

    # Normalize rows to sum to 1 (standard requirement for log loss, though clipping handles extremes)
    probs_ensemble_val = probs_ensemble_val / probs_ensemble_val.sum(
        axis=1, keepdims=True
    )

    val_log_loss = log_loss(
        y_val_split, probs_ensemble_val, labels=np.arange(len(le.classes_))
    )

    print(f"Final Validation Metric: {val_log_loss}")

    # 7. Failure Analysis
    # Get feature names for View 1
    feature_cols = [
        c
        for c in df_train_split.columns
        if any(k in c for k in ["margin", "shape", "texture"])
    ]
    feature_cols.sort()
    perform_failure_analysis(
        X_val_v1, y_val_split, probs_ensemble_val, feature_names=feature_cols
    )

    # =========================================================================
    # PHASE 2: FINAL SUBMISSION (Train on Full Data, Predict on Test)
    # =========================================================================
    THRESHOLD = 0.00870833951594525

    if val_log_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_log_loss}) meets threshold ({THRESHOLD}). Proceeding to submission..."
        )

        # 1. Run Pipeline on Full Data (Train+Val merged, Test)
        print("Running full data pipeline...")
        data = pipeline.run(load_cached_data=True)

        X_train_full_v1 = data["X_train_view1"]
        y_train_full = data["y_train"]
        X_test_v1 = data["X_test_view1"]
        test_ids = data["test_ids"]
        classes = data["classes"]

        # 2. Train Final Models
        print("Training Final Models on Combined Dataset...")

        # Estimator A
        model_a_final = get_discriminative_solver()
        probs_a_test = train_predict(
            model_a_final,
            X_train_full_v1,
            y_train_full,
            X_test_v1,
            model_name="Final_Estimator_A",
        )

        # Estimator B
        model_b_final = get_generative_solver()
        probs_b_test = train_predict(
            model_b_final,
            X_train_full_v1,
            y_train_full,
            X_test_v1,
            model_name="Final_Estimator_B",
        )

        # 3. Ensemble Predictions
        probs_final = (probs_a_test + probs_b_test) / 2.0

        # 4. Post-processing
        # Rescale rows to sum to 1
        row_sums = probs_final.sum(axis=1, keepdims=True)
        probs_final = probs_final / row_sums

        # Clip
        probs_final = clip_probabilities(probs_final)

        # 5. Create Submission File
        print("Creating submission file...")
        submission_df = pd.DataFrame(probs_final, columns=classes)
        submission_df.insert(0, "id", test_ids)

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({val_log_loss}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
