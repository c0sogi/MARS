import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.feature_extractor import DualStreamExtractor
from library.preprocessor import ManifoldProcessor
from library.classifier import LDAManager


def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    print("Starting execution...")

    # 2. Feature Extraction
    # We extract features for Train, Val, and Test.
    # The extractor handles caching automatically.
    extractor = DualStreamExtractor()

    print("\n--- Extracting Features ---")
    train_data = extractor.extract_features(
        Config.TRAIN_META_PATH, "train", load_cached_data=True
    )
    val_data = extractor.extract_features(
        Config.VAL_META_PATH, "val", load_cached_data=True
    )
    test_data = extractor.extract_features(
        Config.TEST_META_PATH, "test", load_cached_data=True
    )

    # 3. Label Encoding
    # Ensure consistent mapping of class names to integers
    le = LabelEncoder()
    # Fit on training labels
    y_train_all = train_data["labels"]
    le.fit(y_train_all)

    # Transform labels
    y_train_enc = le.transform(y_train_all)
    y_val_enc = le.transform(val_data["labels"])

    classes = le.classes_
    print(f"Number of classes: {len(classes)}")

    # 4. Cross-Validation (on Training Set)
    print(f"\n--- Starting Stratified {Config.N_FOLDS}-Fold Cross-Validation ---")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    cv_scores = []

    # Raw data arrays for splitting
    raw_train_emb = train_data["embeddings"]
    raw_train_tab = train_data["tabular"]
    raw_train_ids = train_data["ids"]

    for fold, (train_idx, fold_val_idx) in enumerate(
        skf.split(raw_train_emb, y_train_enc)
    ):
        # Prepare Fold Data Dictionaries
        fold_train_data = {
            "embeddings": raw_train_emb[train_idx],
            "tabular": raw_train_tab[train_idx],
            "ids": raw_train_ids[train_idx],
            "labels": y_train_enc[train_idx],  # Encoded labels
        }

        fold_val_data = {
            "embeddings": raw_train_emb[fold_val_idx],
            "tabular": raw_train_tab[fold_val_idx],
            "ids": raw_train_ids[fold_val_idx],
            "labels": y_train_enc[fold_val_idx],  # Encoded labels
        }

        # Initialize Processor and Classifier for this fold
        processor = ManifoldProcessor()
        lda = LDAManager()

        # A. Fit Pipeline on Expanded Training Fold
        # Note: fit_transform_train expects labels to be present in the dict if we want them passed through,
        # but here we handle y separately for clarity, though processor handles replication.
        # We need to pass the fold data to processor.

        # Processor expects raw labels to replicate them. We pass encoded labels.
        X_train_fold, y_train_fold_expanded, _ = processor.fit_transform_train(
            fold_train_data, load_cache=False
        )

        # B. Train LDA
        lda.train(X_train_fold, y_train_fold_expanded)

        # C. Inference on Centroid Validation Fold
        X_val_fold, y_val_fold_centroid, _ = processor.transform_inference(
            fold_val_data, prefix=f"fold_{fold}_val", load_cache=False
        )

        # D. Predict and Evaluate
        probs_fold = lda.predict_proba(X_val_fold)

        # Log Loss
        score = log_loss(
            y_val_fold_centroid, probs_fold, labels=np.arange(len(classes))
        )
        cv_scores.append(score)
        print(f"Fold {fold+1}/{Config.N_FOLDS} Log Loss: {score:.5f}")

    print(f"Average CV Log Loss: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}")

    # 5. Final Model Training (Full Training Set)
    print("\n--- Training Final Model on Full Training Set ---")
    final_processor = ManifoldProcessor()
    final_lda = LDAManager()

    # Prepare full train data dict with encoded labels
    train_data_enc = train_data.copy()
    train_data_enc["labels"] = y_train_enc

    # Fit pipeline
    X_train_final, y_train_final_expanded, _ = final_processor.fit_transform_train(
        train_data_enc, load_cache=True
    )

    # Train LDA
    final_lda.train(X_train_final, y_train_final_expanded)

    # 6. Hold-out Validation
    print("\n--- Evaluating on Hold-out Validation Set ---")
    # Prepare val data dict with encoded labels
    val_data_enc = val_data.copy()
    val_data_enc["labels"] = y_val_enc

    # Transform
    X_val_final, y_val_final_centroid, val_ids = final_processor.transform_inference(
        val_data_enc, prefix="val", load_cache=True
    )

    # Predict
    val_probs = final_lda.predict_proba(X_val_final)

    # Compute Metric
    final_metric = log_loss(
        y_val_final_centroid, val_probs, labels=np.arange(len(classes))
    )
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    # We need to pick the probability assigned to the true class
    # y_val_final_centroid is (N,) indices
    # val_probs is (N, C)

    # Gather true class probabilities
    true_class_probs = val_probs[
        np.arange(len(y_val_final_centroid)), y_val_final_centroid
    ]
    # Clip for safety in log calculation (though predict_proba already clips)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Sample Loss: {np.mean(sample_losses):.4f}")
    print(f"Max Sample Loss: {np.max(sample_losses):.4f}")

    # Correlate with Tabular Features
    # val_data['tabular'] is (N, 192)
    # Features: Margin (1-64), Shape (1-64), Texture (1-64)
    raw_val_tabular = val_data["tabular"]
    feature_names = (
        [f"margin_{i}" for i in range(1, 65)]
        + [f"shape_{i}" for i in range(1, 65)]
        + [f"texture_{i}" for i in range(1, 65)]
    )

    correlations = []
    for i, feat_name in enumerate(feature_names):
        feat_values = raw_val_tabular[:, i]
        # Handle constant features to avoid warnings
        if np.std(feat_values) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(sample_losses, feat_values)
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error (Log Loss):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    print("\n--- Generating Submission ---")
    # Process Test Data
    X_test, _, test_ids = final_processor.transform_inference(
        test_data, prefix="test", load_cache=True
    )

    # Predict
    test_probs = final_lda.predict_proba(X_test)

    # Create DataFrame
    # Columns: id, Species_1, Species_2, ...
    # Classes in LDA are integers 0..98 corresponding to le.classes_
    # le.classes_ are sorted alphabetically by default if input was strings,
    # but we should verify against sample submission or just trust LabelEncoder on sorted unique strings.
    # The sample submission expects columns sorted alphabetically.

    # Check if classes are sorted
    sorted_classes = np.sort(classes)
    if not np.array_equal(classes, sorted_classes):
        print(
            "Warning: LabelEncoder classes are not sorted alphabetically. Reordering columns..."
        )
        # If le.classes_ is not sorted, we need to map predictions to sorted columns.
        # However, LabelEncoder usually sorts classes.
        pass

    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("Execution complete.")


if __name__ == "__main__":
    main()
