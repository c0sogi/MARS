import os
import numpy as np
import pandas as pd
from library.config import Config
from library.cross_validation import CrossValidator
from library.ensemble_components import get_meta_learner
from library.utils import seed_everything


def main():
    # 1. Setup and Initialization
    # Set fixed random seeds for reproducibility
    seed_everything(Config.SEED)

    # We use debug=False to run on the full dataset.
    # With ~7k images and an A100 GPU, this fits well within the 2-hour limit
    # and is necessary to achieve the target RMSE.
    print("Initializing CrossValidator...")
    cv = CrossValidator(debug=False)

    # 2. Pipeline Execution
    # This orchestrates:
    # - Feature Extraction (Backbones: CLIP, DINOv2, ConvNeXt | Views: Warped, Preserved)
    # - Level-0 Expert Training (Ridge, SVR, ExtraTrees) via 5-Fold CV
    # - Level-1 Meta-Learner Training (Bayesian Ridge)
    # - Submission Generation
    print("Starting pipeline execution...")
    cv_rmse, val_rmse = cv.run()

    # 3. Validation Reporting
    # Print the metric in the exact required format
    print(f"Final Validation Metric: {val_rmse}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    try:
        # The CrossValidator saves Level-0 prediction matrices to disk.
        # We load them to reconstruct the final ensemble predictions for the validation set.
        working_dir = Config.WORKING_DIR

        # Load Level-0 outputs
        l0_oof = np.load(os.path.join(working_dir, "l0_oof.npy"))
        l0_train_targets = np.load(os.path.join(working_dir, "l0_train_targets.npy"))
        l0_val = np.load(os.path.join(working_dir, "l0_val.npy"))
        l0_val_targets = np.load(os.path.join(working_dir, "l0_val_targets.npy"))

        # Instantiate and fit a fresh Meta-Learner (Bayesian Ridge is deterministic given data)
        meta_learner = get_meta_learner(random_state=Config.SEED)
        meta_learner.fit(l0_oof, l0_train_targets)

        # Predict on the validation set Level-0 features
        val_preds = meta_learner.predict(l0_val)

        # Calculate absolute error
        errors = np.abs(l0_val_targets - val_preds)

        # Load validation metadata to analyze correlations
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Verify alignment
        if len(val_df) != len(errors):
            print(
                f"Warning: Metadata length ({len(val_df)}) matches not predictions ({len(errors)}). Skipping correlation analysis."
            )
        else:
            val_df["abs_error"] = errors

            # Calculate correlation between error magnitude and binary features
            print("Correlation between Error Magnitude and Input Features:")
            correlations = {}
            for feature in Config.BINARY_FEATURES:
                if feature in val_df.columns:
                    corr = val_df[feature].corr(val_df["abs_error"])
                    correlations[feature] = corr

            # Sort and print
            sorted_corrs = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for feat, corr in sorted_corrs:
                print(f"{feat}: {corr:.4f}")

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # 5. Submission Management
    # Threshold defined in the task
    THRESHOLD = 17.07053899184464
    submission_path = Config.SUBMISSION_PATH

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmse}) meets threshold ({THRESHOLD}). Submission saved."
        )
    else:
        print(
            f"\nValidation metric ({val_rmse}) does not meet threshold ({THRESHOLD}). Discarding submission."
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
