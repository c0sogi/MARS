import os
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import load_data
from library.features import HybridFeatureGenerator
from library.stacking_manager import StackingManager


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # GPU Optimization for XGBoost
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost to use CUDA.")
        Config.XGB_PARAMS["tree_method"] = "hist"
        Config.XGB_PARAMS["device"] = "cuda"
    else:
        print("No GPU detected. Using CPU for all models.")

    # 2. Data Loading
    print("Loading datasets...")
    train_df, val_df, test_df = load_data()

    # 3. Feature Extraction
    print("Generating features...")
    feature_gen = HybridFeatureGenerator()
    # Process data (handles caching internally)
    data = feature_gen.process(train_df, val_df, test_df, load_cached_data=True)

    # Unpack features
    X_train_sparse = data["train_sparse"]
    X_train_dense = data["train_dense"]
    y_train = data["y_train"]

    X_val_sparse = data["val_sparse"]
    X_val_dense = data["val_dense"]
    y_val = data["y_val"]

    X_test_sparse = data["test_sparse"]
    X_test_dense = data["test_dense"]
    label_classes = data["label_classes"]

    # 4. Stacking Ensemble Execution
    manager = StackingManager()

    # Step A: Generate Out-Of-Fold (OOF) Predictions for Meta-Training
    # This performs K-Fold CV on the training set
    oof_preds = manager.get_oof_predictions(
        X_train_sparse, X_train_dense, y_train, load_cached_data=True
    )

    # Step B: Train Meta-Learner on OOF predictions
    meta_learner = manager.train_meta_learner(oof_preds, y_train)

    # Step C: Validation Inference
    # To get a valid validation score, we train base models on the full Training set
    # and predict on the Validation set.
    print("Refitting base models on Train set for validation...")
    base_models_val = manager.refit_base_models(X_train_sparse, X_train_dense, y_train)

    print("Predicting on Validation set...")
    val_probs = manager.predict_ensemble(
        base_models_val, meta_learner, X_val_sparse, X_val_dense
    )

    # 5. Evaluation
    # Calculate Log Loss
    val_loss = calculate_log_loss(y_val, val_probs)
    print(f"Final Validation Metric: {val_loss}")

    # Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate error magnitude (1 - probability of the true class)
    # y_val is integer encoded, val_probs is (N_samples, N_classes)
    prob_true_class = val_probs[np.arange(len(y_val)), y_val]
    error_magnitude = 1.0 - prob_true_class

    # Get text lengths for correlation analysis
    # Fill NaN with empty string to be safe, though preprocessing usually handles this
    val_text = val_df["text"].fillna("").astype(str)
    char_counts = val_text.apply(len).values
    word_counts = val_text.apply(lambda x: len(x.split())).values

    # Compute correlations
    corr_char = np.corrcoef(error_magnitude, char_counts)[0, 1]
    corr_word = np.corrcoef(error_magnitude, word_counts)[0, 1]

    print(f"Correlation (Error vs Char Count): {corr_char:.6f}")
    print(f"Correlation (Error vs Word Count): {corr_word:.6f}")

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.34671256711662346

    if val_loss < THRESHOLD:
        print(
            f"\nValidation score ({val_loss:.6f}) meets threshold ({THRESHOLD}). Proceeding to submission."
        )

        # For the final submission, we refit base models on ALL available labeled data (Train + Val)
        # to maximize performance. We reuse the meta-learner trained on OOF.
        print("Refitting base models on combined Train + Validation data...")

        X_all_sparse = scipy.sparse.vstack([X_train_sparse, X_val_sparse])
        X_all_dense = np.vstack([X_train_dense, X_val_dense])
        y_all = np.concatenate([y_train, y_val])

        final_base_models = manager.refit_base_models(X_all_sparse, X_all_dense, y_all)

        print("Generating predictions for Test set...")
        test_probs = manager.predict_ensemble(
            final_base_models, meta_learner, X_test_sparse, X_test_dense
        )

        # Save submission
        test_ids = test_df["id"].values
        manager.save_submission(test_ids, test_probs, label_classes)

    else:
        print(
            f"\nValidation score ({val_loss:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
