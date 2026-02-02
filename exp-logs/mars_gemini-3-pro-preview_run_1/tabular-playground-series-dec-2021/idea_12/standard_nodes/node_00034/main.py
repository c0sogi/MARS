import os
import gc
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.features import preprocess_data
from library.model import XGBoostTrainer, inject_knn_features


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Starting Fast Baseline Run...")

    # 2. Load Data
    # Load full data (debug=False)
    # We utilize the full dataset as per Lesson solution_lesson_node_00002
    train_df, val_df, test_df = preprocess_data(load_cached_data=True, debug=False)

    logger.info(f"Training Shape: {train_df.shape}")
    logger.info(f"Validation Shape: {val_df.shape}")

    # 3. Target Encoding
    le = LabelEncoder()
    train_df[Config.TARGET_COL] = le.fit_transform(train_df[Config.TARGET_COL])
    # Transform Val using the same encoder
    val_df[Config.TARGET_COL] = le.transform(val_df[Config.TARGET_COL])
    num_classes = len(le.classes_)
    logger.info(f"Encoded Target. Classes: {num_classes}")

    # 4. Feature Engineering: Inject Manifold-Aware k-NN Features
    # To scale to the full dataset, we use a "Landmark" strategy.
    # We sample a subset of the training data to serve as the reference manifold.
    # This avoids O(N^2) complexity while preserving density information.
    LANDMARK_SIZE = 100000
    logger.info(f"Sampling {LANDMARK_SIZE} landmarks for k-NN reference...")
    ref_df = train_df.sample(
        n=min(len(train_df), LANDMARK_SIZE), random_state=Config.SEED
    )

    # A. Train Set (Ref=Landmarks, Exclude Self=False)
    # We disable exclude_self because the query set (Full Train) contains the Reference set.
    # While this means dist=0 for the landmarks themselves, it's a consistent signal.
    logger.info("Injecting k-NN features into Training set...")
    train_aug = inject_knn_features(
        ref_df=ref_df,
        query_df=train_df,
        knn_cols=Config.KNN_FEATURES,
        exclude_self=False,
    )

    # B. Validation Set (Ref=Landmarks, Exclude Self=False)
    logger.info("Injecting k-NN features into Validation set...")
    val_aug = inject_knn_features(
        ref_df=ref_df,
        query_df=val_df,
        knn_cols=Config.KNN_FEATURES,
        exclude_self=False,
    )

    # Define Feature Columns (exclude ID, Target, and raw Scaled columns used for KNN)
    ignore_cols = [Config.ID_COL, Config.TARGET_COL] + [
        f"{c}_scaled" for c in Config.KNN_FEATURES
    ]
    feature_cols = [c for c in train_aug.columns if c not in ignore_cols]

    logger.info(f"Using {len(feature_cols)} features.")

    # Prepare matrices
    X_train = train_aug[feature_cols]
    y_train = train_aug[Config.TARGET_COL]
    X_val = val_aug[feature_cols]
    y_val = val_aug[Config.TARGET_COL]

    # 5. Model Training
    logger.info("Initializing XGBoost Trainer...")
    trainer = XGBoostTrainer(num_class=num_classes)

    logger.info("Fitting model...")
    # Config.NUM_BOOST_ROUND is 5000, but early stopping (50) will optimize runtime
    model = trainer.fit(X_train, y_train, X_val, y_val)

    # 6. Validation Inference
    logger.info("Predicting on Validation set...")
    val_probs = trainer.predict_proba(model, X_val)
    val_preds = np.argmax(val_probs, axis=1)

    # 7. Metric Calculation
    acc = accuracy_score(y_val, val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {acc}")

    # 8. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Error flag: 1 if prediction is wrong, 0 if correct
    errors = (val_preds != y_val).astype(int)

    # Create analysis dataframe
    analysis_df = X_val.copy()
    analysis_df["Error_Flag"] = errors

    # Calculate correlation of features with Error_Flag
    # We use abs() to find strong relationships regardless of direction
    corrs = (
        analysis_df.corrwith(analysis_df["Error_Flag"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop Features Correlated with Error:")
    # Skip the first one if it is Error_Flag itself (corr=1.0)
    print(corrs.head(6))

    # 9. Submission
    THRESHOLD = 0.9628180555555556

    if acc > THRESHOLD:
        logger.info(
            f"Validation accuracy {acc} > {THRESHOLD}. Generating submission..."
        )

        # Inject k-NN features into Test set (Ref=Landmarks)
        # We use ref_df (Landmarks) as reference for consistency with Train/Val
        logger.info("Injecting k-NN features into Test set...")
        test_aug = inject_knn_features(
            ref_df=ref_df,
            query_df=test_df,
            knn_cols=Config.KNN_FEATURES,
            exclude_self=False,
        )

        X_test = test_aug[feature_cols]

        logger.info("Predicting on Test set...")
        test_probs = trainer.predict_proba(model, X_test)
        test_preds = np.argmax(test_probs, axis=1)

        # Inverse transform labels
        final_preds = le.inverse_transform(test_preds)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        logger.info(f"Validation accuracy {acc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
