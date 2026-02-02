import os
import numpy as np
import pandas as pd
import torch
from library import config, utils, feature_extraction, preprocessing, modeling


def main():
    # 1. Setup and Reproducibility
    print("Initializing...")
    utils.seed_everything(config.SEED)

    # Ensure working directory exists for caching
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Feature Extraction
    # We extract features for Train, Validation, and Test sets.
    # The extract_features function handles caching automatically.
    # If cache exists in ./working/idea_4/, it loads it; otherwise it runs the models.
    print("Extracting features (this may take a moment if not cached)...")

    # Train set
    train_cnn, train_vit, train_tab, train_y, train_ids = (
        feature_extraction.extract_features(split="train", load_cached_data=True)
    )

    # Validation set
    val_cnn, val_vit, val_tab, val_y, val_ids = feature_extraction.extract_features(
        split="val", load_cached_data=True
    )

    # Test set
    test_cnn, test_vit, test_tab, _, test_ids = feature_extraction.extract_features(
        split="test", load_cached_data=True
    )

    print(f"Feature Extraction Complete.")
    print(
        f"Train shapes: CNN {train_cnn.shape}, ViT {train_vit.shape}, Tabular {train_tab.shape}"
    )

    # 3. Preprocessing
    print("Preprocessing features...")

    # A. Tabular Data: Apply Quantile Transformer (Gaussianization)
    # This helps LDA which assumes Gaussian distributed classes.
    tab_proc = preprocessing.TabularGaussianizer(n_quantiles=min(len(train_tab), 1000))
    tab_proc.fit(train_tab)

    train_tab_proc = tab_proc.transform(train_tab)
    val_tab_proc = tab_proc.transform(val_tab)
    test_tab_proc = tab_proc.transform(test_tab)

    # B. Embeddings: Apply PCA
    # Reduce dimensionality of high-dim embeddings (CNN: 1536, ViT: 1024)

    # CNN PCA
    cnn_reducer = preprocessing.EmbeddingReducer(n_components=config.PCA_VARIANCE)
    cnn_reducer.fit(train_cnn)

    train_cnn_proc = cnn_reducer.transform(train_cnn)
    val_cnn_proc = cnn_reducer.transform(val_cnn)
    test_cnn_proc = cnn_reducer.transform(test_cnn)

    print(
        f"CNN PCA: Reduced from {train_cnn.shape[1]} to {train_cnn_proc.shape[1]} components."
    )

    # ViT PCA
    vit_reducer = preprocessing.EmbeddingReducer(n_components=config.PCA_VARIANCE)
    vit_reducer.fit(train_vit)

    train_vit_proc = vit_reducer.transform(train_vit)
    val_vit_proc = vit_reducer.transform(val_vit)
    test_vit_proc = vit_reducer.transform(test_vit)

    print(
        f"ViT PCA: Reduced from {train_vit.shape[1]} to {train_vit_proc.shape[1]} components."
    )

    # 4. Feature Concatenation
    # Combine processed tabular, CNN, and ViT features into a single feature vector
    X_train = np.concatenate([train_tab_proc, train_cnn_proc, train_vit_proc], axis=1)
    X_val = np.concatenate([val_tab_proc, val_cnn_proc, val_vit_proc], axis=1)
    X_test = np.concatenate([test_tab_proc, test_cnn_proc, test_vit_proc], axis=1)

    print(f"Final Feature Vector Shape: {X_train.shape}")

    # 5. Modeling
    print("Training Hybrid Ensemble (LDA + Logistic Regression)...")
    ensemble = modeling.HybridEnsemble(random_state=config.SEED)

    # Fit models on training data
    ensemble.fit_models(X_train, train_y)

    # Optimize mixing weight on validation data
    # This finds the best balance between LDA and LR based on Log Loss
    best_weight = ensemble.find_optimal_weight(X_val, val_y)

    # 6. Prediction & Submission
    print("Generating predictions for test set...")
    test_probs = ensemble.predict_proba(X_test)

    # Retrieve class names
    # The data_loader caches class names during loading
    classes_path = os.path.join(config.WORKING_DIR, "classes.npy")
    if os.path.exists(classes_path):
        class_names = np.load(classes_path, allow_pickle=True)
    else:
        # Fallback: derive from train metadata if not cached
        df_train = pd.read_csv(config.TRAIN_META_PATH)
        class_names = np.sort(df_train["species"].unique())

    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    utils.save_submission(test_ids, class_names, test_probs, config.SUBMISSION_PATH)

    # 7. Verification Steps
    print("Verifying outputs...")

    # A. Check submission file existence
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    # B. Check submission format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Check row count (should match test set size)
    expected_rows = len(test_ids)
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission has {len(df_sub)} rows, expected {expected_rows}."
        )

    # Check column count (id + 99 classes)
    expected_cols = 1 + 99
    if len(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Submission has {len(df_sub.columns)} columns, expected {expected_cols}."
        )

    # Check ID alignment
    if not np.array_equal(df_sub["id"].values, test_ids):
        raise AssertionError("Submission IDs do not match Test IDs.")

    # C. Check Probability Range
    probs_only = df_sub.drop(columns=["id"]).values
    if probs_only.min() < 0 or probs_only.max() > 1:
        raise AssertionError("Probabilities out of range [0, 1].")

    print("Verification successful. Pipeline executed correctly.")


if __name__ == "__main__":
    main()
