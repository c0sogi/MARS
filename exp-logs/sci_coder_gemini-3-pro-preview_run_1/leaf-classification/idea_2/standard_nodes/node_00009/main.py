import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.utils import set_seed, score_predictions
from library.data_loader import LeafDataManager
from library.models import LDAWrapper


def main():
    # 1. Setup and Reproducibility
    set_seed(42)
    print("Initializing Fast Baseline Run...")

    # 2. Data Loading
    # We need two views of the data:
    # 'tree' (raw) for LightGBM
    # 'linear_kernel' (transformed) for LDA and SVM
    dm = LeafDataManager(seed=42)

    print("Loading datasets...")
    # Train Data
    # Cite solution_lesson_node_00006: We focus on LDA with transformed features.
    X_train_lin, y_train = dm.get_train_data(model_type="linear_kernel")

    # Validation Data
    # We need tree data for failure analysis (interpretability) and linear for prediction
    X_val_tree, y_val = dm.get_val_data(model_type="tree")
    X_val_lin, _ = dm.get_val_data(model_type="linear_kernel")

    # Test Data
    # Only need linear features for LDA inference
    X_test_lin, test_ids = dm.get_test_data(model_type="linear_kernel")

    # Classes
    classes = dm.get_classes()
    class_indices = np.arange(len(classes))

    # 3. Model Training
    print("Training models...")

    # Cite solution_lesson_node_00006: Enforcing Normality via Power Transforms for LDA.
    # LDA on PowerTransformed features significantly outperforms GBDT and SVM on this dataset.
    # We rely solely on LDA to avoid ensemble noise and reduce runtime.

    # Training LDA (Linear)
    print("  Training LDA...")
    model_lda = LDAWrapper(random_state=42)
    model_lda.fit(X_train_lin, y_train, X_val=X_val_lin, y_val=y_val)

    # 4. Validation Inference
    # Get probabilities on validation set
    final_val_preds = model_lda.predict_proba(X_val_lin)

    # 5. Validation Metric
    # Calculate score using the provided utility which handles clipping/rescaling
    val_score = score_predictions(y_val, final_val_preds, classes=class_indices)
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (Log Loss per sample)
    # Clip predictions to avoid log(0)
    eps = 1e-15
    preds_clipped = np.clip(final_val_preds, eps, 1 - eps)
    # Normalize rows
    preds_norm = preds_clipped / preds_clipped.sum(axis=1, keepdims=True)

    # Extract probability assigned to the true class
    # y_val contains indices of true classes
    row_indices = np.arange(len(y_val))
    true_class_probs = preds_norm[row_indices, y_val]

    # Error magnitude = -log(p_true)
    error_magnitudes = -np.log(true_class_probs)

    # Correlate error with input features (using raw tree features for interpretability)
    # We need feature names
    df_train_meta = pd.read_csv("./metadata/train.csv")
    feature_names = [
        c for c in df_train_meta.columns if c.startswith(("margin", "shape", "texture"))
    ]

    correlations = []
    for i in range(X_val_tree.shape[1]):
        feat_values = X_val_tree[:, i]
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_values, error_magnitudes)
        correlations.append(corr)

    correlations = np.array(correlations)
    # Get top 5 absolute correlations
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 features correlated with error magnitude:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {correlations[idx]:.4f}")

    # 7. Submission Generation
    TARGET_THRESHOLD = 3.3960549710866674e-07

    if val_score < TARGET_THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        final_test_preds = model_lda.predict_proba(X_test_lin)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(final_test_preds, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    else:
        print(
            f"\nValidation score ({val_score}) does NOT meet threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
