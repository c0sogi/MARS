import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_loader import LeafDataManager
from library.feature_engine import DualStreamExtractor
from library.cv_runner import StratifiedEnsembleTrainer
from library.inference_engine import EnsemblePredictor


def main():
    # 1. Initialization and Configuration
    print("Initializing workflow...")
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Loading (Training)
    print("\n--- Data Loading (Train) ---")
    dm = LeafDataManager()
    # Load tabular data, labels, paths, and IDs
    # load_cached_data=True allows using pre-computed .npy files if they exist
    X_tab_train, y_train, train_paths, train_ids = dm.load_train_data(
        load_cached_data=True
    )

    # 3. Feature Extraction (Training)
    print("\n--- Feature Extraction (Train) ---")
    extractor = DualStreamExtractor()
    # Extract or load DINOv2 and ConvNeXt features
    dino_train, conv_train = extractor.get_train_features(load_cached_data=True)

    # 4. Model Training & Validation
    print("\n--- Model Training & Validation ---")
    trainer = StratifiedEnsembleTrainer()
    # Run Stratified K-Fold CV
    # This trains the ensemble, saves pipelines, and prints fold scores
    fold_scores = trainer.cross_validate(dino_train, conv_train, X_tab_train, y_train)

    # Compute and print the final validation metric
    final_metric = np.mean(fold_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # To analyze failure, we need OOF predictions.
    # Since the trainer saves models but doesn't return OOF preds, we reconstruct them here.
    # This is fast as inference is cheap compared to feature extraction.

    # Concatenate features to match the pipeline input format [DINO | CONV | TABULAR]
    X_concat = np.hstack([dino_train, conv_train, X_tab_train])
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    oof_probs = np.zeros((n_samples, n_classes))

    # Re-create splits to match training
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    print("Generating OOF predictions for analysis...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_concat, y_train)):
        # Load the saved pipeline for this fold
        pipeline_path = Config.PIPELINE_PATH.format(fold=fold)
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        # Predict on validation set
        X_val = X_concat[val_idx]
        oof_probs[val_idx] = pipeline.predict_proba(X_val)

    # Calculate Error Magnitude
    # Error = 1.0 - Probability assigned to the true class
    # y_train contains class indices
    row_indices = np.arange(n_samples)
    true_class_probs = oof_probs[row_indices, y_train]
    error_magnitude = 1.0 - true_class_probs

    # Correlate Error Magnitude with Tabular Features
    # We need feature names to make this meaningful
    # Re-read metadata to get column names
    df_meta = pd.read_csv(Config.TRAIN_METADATA, nrows=1)
    feature_names = []
    for prefix in Config.TABULAR_PREFIXES:
        cols = [c for c in df_meta.columns if c.startswith(prefix)]
        feature_names.extend(cols)
    feature_names.sort()  # Ensure order matches DataManager

    # Compute correlations
    print("Computing correlations between features and error magnitude...")
    correlations = {}
    for i, name in enumerate(feature_names):
        # X_tab_train column i corresponds to feature_names[i]
        if np.std(X_tab_train[:, i]) > 0:  # Avoid constant features
            corr = np.corrcoef(X_tab_train[:, i], error_magnitude)[0, 1]
            correlations[name] = corr
        else:
            correlations[name] = 0.0

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Associated with Error (High Absolute Correlation):")
    for name, corr in sorted_corr[:10]:
        print(f"  {name}: {corr:.4f}")

    # 6. Inference & Submission
    print("\n--- Inference & Submission ---")

    # Check metric threshold condition
    # Note: The prompt specifies a threshold of ~2.22e-16 (machine epsilon).
    # Strictly following this would likely prevent submission.
    # We proceed with submission to fulfill the "Submit a csv file" requirement.
    threshold = 2.2204460492503136e-16
    if final_metric < threshold:
        print(f"Metric {final_metric} meets the strict threshold (< {threshold}).")
    else:
        print(
            f"Metric {final_metric} does not meet the strict threshold (< {threshold}). Proceeding with submission anyway."
        )

    # Load Test Data
    X_tab_test, test_ids, test_paths = dm.load_test_data(load_cached_data=True)

    # Extract Test Features
    dino_test, conv_test = extractor.get_test_features(load_cached_data=True)

    # Generate Submission
    predictor = EnsemblePredictor()
    predictor.create_submission(dino_test, conv_test, X_tab_test, test_ids)

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
