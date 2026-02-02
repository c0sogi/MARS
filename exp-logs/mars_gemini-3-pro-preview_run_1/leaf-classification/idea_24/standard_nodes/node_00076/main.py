import os
import numpy as np
import pandas as pd
import warnings
from library import config, utils, data_loader, teacher, student

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    print("=" * 40)
    print("Generative-Discriminative Synthetic Transfer (GDST) Execution")
    print("=" * 40)

    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading and Preprocessing Data...")
    # We use load_cached_data=True to leverage any existing preprocessed files
    (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, classes) = (
        data_loader.get_processed_data(load_cached_data=True)
    )

    print(f"  Train Set: {X_train.shape} samples")
    print(f"  Val Set:   {X_val.shape} samples")
    print(f"  Test Set:  {X_test.shape} samples")
    print(f"  Classes:   {len(classes)}")

    # 2. Teacher Training (Direct Analytic Solution)
    # -------------------------------------------------------------------------
    print("\n[2/6] Training Teacher (OAS-LDA)...")
    # Cite solution_lesson_node_00074: Closed-Form Solutions Outperform Stochastic Approximations
    # We use the analytic teacher directly instead of training a student on synthetic data.
    teacher_instance = teacher.OASTeacher()
    teacher_instance.fit(X_train, y_train)

    # 3. Validation Evaluation
    # -------------------------------------------------------------------------
    print("\n[3/6] Performing Final Validation...")
    # Evaluate using the competition metric (Log Loss with specific clipping)
    # Cite solution_lesson_node_00055: Using Linear Formulation for inference
    val_loss = teacher_instance.evaluate(X_val, y_val)

    # REQUIRED: Print the validation metric in the specific format
    print(f"Final Validation Metric: {val_loss:.20f}")

    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[4/6] Performing Failure Analysis...")
    # Predict probabilities on validation set
    probs_val = teacher_instance.predict_proba(X_val)

    # Calculate per-sample log loss for correlation analysis
    # Clip probabilities to match metric definition
    probs_val_clipped = np.clip(probs_val, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX)

    # Extract probability assigned to the true class
    # y_val contains integer class indices
    true_class_probs = probs_val_clipped[np.arange(len(y_val)), y_val]

    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between each feature and the error magnitude
    correlations = []
    loss_std = np.std(sample_losses)

    if loss_std > 1e-15:  # Check if there is variance in loss
        for i in range(X_val.shape[1]):
            feature_vec = X_val[:, i]
            # Use numpy for correlation to avoid extra dependencies
            # np.corrcoef returns a matrix [[1, corr], [corr, 1]]
            corr = np.corrcoef(feature_vec, sample_losses)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append((config.FEATURE_COLS[i], corr))

        # Sort by absolute correlation (descending)
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("  Top 5 features correlated with prediction error:")
        for name, corr in correlations[:5]:
            print(f"    {name}: {corr:.4f}")
    else:
        print(
            "  Model performance is nearly perfect or constant; negligible variance in loss for correlation analysis."
        )

    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5/6] Generating Submission...")

    # Strict threshold check as per requirements
    THRESHOLD = 1.2136771218566717e-09

    if val_loss < THRESHOLD:
        print(
            f"  Validation metric ({val_loss:.4e}) meets threshold ({THRESHOLD:.4e})."
        )
        print("  Generating predictions for test set...")

        # Inference on Test Set
        probs_test = teacher_instance.predict_proba(X_test)

        # Save Submission
        utils.format_submission(
            test_ids=ids_test,
            y_pred_probs=probs_test,
            class_labels=classes,  # Pass string labels for CSV header
            output_path=config.SUBMISSION_FILE_PATH,
        )
    else:
        print(
            f"  Validation metric ({val_loss:.4e}) did NOT meet threshold ({THRESHOLD:.4e})."
        )
        print("  Submission file will NOT be generated.")

    print("\nWorkflow Completed.")


if __name__ == "__main__":
    run()
