import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge

from library.config import Config
from library.utils import seed_everything
from library.feature_extraction import extract_and_save_features
from library.level1_meta import MetaLearner


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Feature Extraction
    # We explicitly run this first to ensure all features are cached and available
    # for the stacking process. We use a slightly conservative batch size for CLIP.
    print("=== Starting Feature Extraction ===")

    backbones = [Config.MODEL_CLIP, Config.MODEL_DINO, Config.MODEL_CONVNEXT]
    splits = ["train", "val", "test"]

    for model_name in backbones:
        # Adjust batch size for larger CLIP model to ensure safety on GPU
        b_size = 32 if model_name == Config.MODEL_CLIP else 64

        for split in splits:
            extract_and_save_features(
                model_name=model_name,
                split=split,
                load_cached_data=True,
                batch_size=b_size,
            )

    # 3. Meta Learner Execution
    # This handles Level-0 training (if not cached), Meta-Feature generation,
    # and creates the submission.csv based on the full training data.
    print("\n=== Executing Meta Learner Pipeline ===")
    learner = MetaLearner()
    learner.run(load_cached_data=True)

    # 4. Hold-out Validation
    # To strictly satisfy the requirement of validating on the hold-out set,
    # we reconstruct the split from the meta-features.
    print("\n=== Performing Hold-out Validation ===")

    # Load original metadata to get split lengths
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    n_train = len(train_df)

    # Get Meta-Features (Stacked OOF predictions from L0 experts)
    # The Level0Trainer concatenates Train and Val sets, so we can slice them back.
    X_meta, y_meta, _, _ = learner.get_meta_features(load_cached_data=True)

    # Slice into Train and Validation
    X_train_meta = X_meta[:n_train]
    y_train_meta = y_meta[:n_train]
    X_val_meta = X_meta[n_train:]
    y_val_meta = y_meta[n_train:]

    # Train L1 Model on Train split ONLY
    l1_validator = BayesianRidge(max_iter=Config.META_N_ITER, tol=Config.META_TOL)
    l1_validator.fit(X_train_meta, y_train_meta)

    # Predict on Hold-out Validation split
    val_preds = l1_validator.predict(X_val_meta)

    # Calculate and Print Metric
    val_rmse = np.sqrt(mean_squared_error(y_val_meta, val_preds))
    print(f"Final Validation Metric: {val_rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    errors = np.abs(y_val_meta - val_preds)

    # Metadata columns to analyze
    meta_cols = [
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

    correlations = {}
    print("Correlation between Absolute Error and Metadata Features:")
    for col in meta_cols:
        if col in val_df.columns:
            # Calculate correlation (handle potential constant columns gracefully)
            if val_df[col].nunique() > 1:
                corr = np.corrcoef(val_df[col].values, errors)[0, 1]
            else:
                corr = 0.0
            correlations[col] = corr

    # Sort by correlation magnitude (descending)
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for col, corr in sorted_corrs:
        print(f"{col}: {corr:.6f}")

    # 6. Submission Condition Check
    # Threshold defined in task
    THRESHOLD = 17.07053899184464

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation score ({val_rmse}) meets threshold ({THRESHOLD}). Submission preserved."
        )
    else:
        print(
            f"\nValidation score ({val_rmse}) does not meet threshold ({THRESHOLD}). Removing submission."
        )
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
