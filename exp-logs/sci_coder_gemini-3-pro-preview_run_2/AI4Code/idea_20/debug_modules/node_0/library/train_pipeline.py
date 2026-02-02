import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_loader import load_notebooks
from library.vectorizer import TextVectorizer
from library.feature_extractor import AnchorFeatureGenerator
from library.model_wrapper import Stage1Ridge, Stage2LGBM


def run_training(debug: bool = False, load_cached_data: bool = True):
    """
    Executes the complete two-stage training pipeline for the cell ordering task.

    Pipeline Steps:
    1. Load Train and Validation data.
    2. Fit or Load TextVectorizer (TF-IDF + SVD).
    3. Train Stage 1 Ridge Regression and generate OOF predictions.
    4. Extract Multi-Resolution Anchor Features (Lexical + Latent).
    5. Construct Stage 2 Dataset (Features + Stage 1 Preds).
    6. Train Stage 2 LightGBM model.
    7. Save all artifacts.

    Args:
        debug (bool): If True, subsamples the dataset for rapid testing.
        load_cached_data (bool): If True, attempts to load intermediate parquet/npy/joblib files
                                 from the working directory to save time.
    """
    # Initialize configuration and ensure directories exist
    Config.setup()

    print(f"Starting Training Pipeline (Debug={debug}, Cache={load_cached_data})")

    # --------------------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------------------
    print("\n[Step 1/6] Loading Notebook Data...")
    train_df = load_notebooks("train", load_cached_data=load_cached_data, debug=debug)
    val_df = load_notebooks("val", load_cached_data=load_cached_data, debug=debug)

    # --------------------------------------------------------------------------
    # 2. Vectorization (TF-IDF + SVD)
    # --------------------------------------------------------------------------
    print("\n[Step 2/6] Fitting/Loading Text Vectorizer...")
    vectorizer = TextVectorizer()
    vec_base_path = os.path.join(Config.WORKING_DIR, "text_vectorizer")

    # Manual caching logic for Vectorizer since it's not handled by a helper function
    tfidf_exists = os.path.exists(f"{vec_base_path}_tfidf.joblib")
    svd_exists = os.path.exists(f"{vec_base_path}_svd.joblib")

    if load_cached_data and tfidf_exists and svd_exists:
        vectorizer.load(vec_base_path)
    else:
        # Fit on all text in training set (Code + Markdown) to build full vocabulary
        # Fill NaNs to ensure robustness
        print("Fitting vectorizer on full training corpus...")
        all_text = train_df["source"].fillna("").astype(str)
        vectorizer.fit(all_text)
        vectorizer.save(vec_base_path)

    # --------------------------------------------------------------------------
    # 3. Stage 1: Ridge Regression (Sparse Lexical)
    # --------------------------------------------------------------------------
    print("\n[Step 3/6] Stage 1 - Ridge Regression...")
    ridge_model = Stage1Ridge()

    # Filter for Markdown cells (Targets) as we only predict rank for markdown
    train_md = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
    val_md = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

    # Transform to Sparse TF-IDF
    print("Transforming text to sparse TF-IDF features...")
    X_train_sparse = vectorizer.transform(train_md["source"].fillna("").astype(str))
    y_train = train_md["norm_rank"].values

    X_val_sparse = vectorizer.transform(val_md["source"].fillna("").astype(str))

    # Generate OOF Predictions for Train (Features for Stage 2)
    # This method also fits the final Ridge model on all training data
    print("Generating Stage 1 OOF predictions...")
    train_oof_preds = ridge_model.get_oof_predictions(
        X_train_sparse, y_train, load_cached_data=load_cached_data
    )

    # Generate Predictions for Validation (Features for Stage 2)
    print("Generating Stage 1 Validation predictions...")
    val_preds = ridge_model.predict(X_val_sparse)

    # Attach predictions to the MD dataframes for merging later
    train_md["stage1_pred"] = train_oof_preds
    val_md["stage1_pred"] = val_preds

    # --------------------------------------------------------------------------
    # 4. Feature Extraction (Multi-Resolution Anchors)
    # --------------------------------------------------------------------------
    print("\n[Step 4/6] Extracting Multi-Resolution Anchor Features...")
    feature_gen = AnchorFeatureGenerator(vectorizer)

    # Extract features (returns DataFrame with 'id', 'cell_id', 'norm_rank', and features)
    # Note: extract_features processes the context of the whole notebook but returns rows for markdown cells
    train_features = feature_gen.extract_features(
        train_df, "train", load_cached_data=load_cached_data
    )
    val_features = feature_gen.extract_features(
        val_df, "val", load_cached_data=load_cached_data
    )

    # --------------------------------------------------------------------------
    # 5. Prepare Stage 2 Datasets
    # --------------------------------------------------------------------------
    print("\n[Step 5/6] Preparing Stage 2 Datasets...")

    # Merge Stage 1 predictions into the feature DataFrames
    # We merge on ['id', 'cell_id'] to ensure correct alignment between the OOF preds and the anchor features
    train_final = train_features.merge(
        train_md[["id", "cell_id", "stage1_pred"]], on=["id", "cell_id"], how="left"
    )

    val_final = val_features.merge(
        val_md[["id", "cell_id", "stage1_pred"]], on=["id", "cell_id"], how="left"
    )

    # Identify Feature Columns
    # Exclude metadata and target columns
    exclude_cols = [
        "id",
        "cell_id",
        "norm_rank",
        "cell_type",
        "source",
        "ancestor_id",
        "parent_id",
    ]
    feature_cols = [c for c in train_final.columns if c not in exclude_cols]

    print(f"Number of Stage 2 features: {len(feature_cols)}")

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM (Refinement)
    # --------------------------------------------------------------------------
    print("\n[Step 6/6] Stage 2 - LightGBM Training...")
    lgbm_model = Stage2LGBM()

    lgbm_model.fit(
        train_df=train_final,
        val_df=val_final,
        feature_cols=feature_cols,
        target_col="norm_rank",
    )

    # Save the trained LightGBM model
    lgbm_path = os.path.join(Config.WORKING_DIR, "stage2_lgbm")
    lgbm_model.save(lgbm_path)

    print("\nTraining pipeline completed successfully.")
