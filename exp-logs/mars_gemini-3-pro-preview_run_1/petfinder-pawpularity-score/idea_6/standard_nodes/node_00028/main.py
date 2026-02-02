import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.engine import StackingTrainer
from library.models import ModelFactory


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # Initialize the trainer engine
    trainer = StackingTrainer()

    # Containers for Level-1 Stacking
    level1_oof_features = []
    level1_test_features = []
    y_target = None

    print("Starting Tri-Paradigm Stacking Pipeline...")

    # 2. Level-0 Experts: Feature Extraction & Base Model Training
    # We iterate through defined backbones (CLIP, DINOv2, ConvNeXt)
    # process_backbone handles feature extraction, caching, and 5-fold CV for Ridge/SVR/ExtraTrees
    for backbone in Config.BACKBONES:
        # load_cached_data=True ensures we use pre-computed features if available
        # debug=False ensures we use the full dataset to achieve the target score
        oof, test, y = trainer.process_backbone(
            backbone, load_cached_data=True, debug=False
        )

        level1_oof_features.append(oof)
        level1_test_features.append(test)

        # Verify target consistency across backbones
        if y_target is None:
            y_target = y
        else:
            if not np.allclose(y_target, y):
                raise ValueError(f"Target mismatch detected for backbone {backbone}.")

    # Stack predictions: (N_samples, 3_backbones * 3_experts) -> (N, 9)
    X_meta_train = np.hstack(level1_oof_features)
    X_meta_test = np.hstack(level1_test_features)

    # 3. Level-1 Meta-Learner Training
    # Train Ridge Regression on the OOF predictions of the base experts
    meta_learner = ModelFactory.get_meta_learner()
    meta_learner.fit(X_meta_train, y_target)

    # Generate final OOF predictions for validation
    oof_final_preds = meta_learner.predict(X_meta_train)

    # 4. Validation Metric
    # Calculate RMSE on the full Train+Val set (Cross-Validation performance)
    # Note: In this pipeline, the "Validation Metric" represents the OOF RMSE,
    # which is the standard estimate of generalization error in stacking.
    final_rmse = compute_rmse(y_target, oof_final_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Failure Analysis
    # Analyze correlation between error magnitude and binary metadata features
    print("\nPerforming Failure Analysis...")

    # Calculate absolute errors
    errors = np.abs(y_target - oof_final_preds)

    # Load metadata to match the order of y_target (Train then Val)
    # The engine concatenates train then val, so we must do the same
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Define binary features
    binary_features = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # Calculate correlations
    correlations = {}
    for feature in binary_features:
        if feature in df_full.columns:
            # Point-biserial correlation equivalent since features are binary
            corr = np.corrcoef(df_full[feature].values, errors)[0, 1]
            correlations[feature] = corr

    # Print correlations sorted by magnitude
    print("Correlation between Error Magnitude and Metadata Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"{feat}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 17.07053899184464

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({final_rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on test set using the meta-learner
        final_test_preds = meta_learner.predict(X_meta_test)

        # Load Test IDs
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        ids_test = df_test["Id"].values

        # Create submission DataFrame
        submission = pd.DataFrame({"Id": ids_test, "Pawpularity": final_test_preds})

        # Save to disk
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_rmse}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
