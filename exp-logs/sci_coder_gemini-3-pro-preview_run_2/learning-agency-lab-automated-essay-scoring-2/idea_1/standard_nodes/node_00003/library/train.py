import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_data
from library.features import extract_features
from library.model import ScoreRegressor
from library.metrics import compute_qwk


def run_training(load_cached_data: bool = True, nrows: int = None):
    """
    Orchestrates the training pipeline:
    1. Loads train/val data.
    2. Extracts TF-IDF features.
    3. Trains Ridge Regression with Alpha tuning.
    4. Evaluates on Validation set.
    5. Saves the model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        nrows (int, optional): Number of rows to load for debugging.
    """
    # 1. Setup
    seed_everything()
    Config.create_dirs()

    print("--- Starting Training Pipeline ---")

    # 2. Load Data
    print("Loading Training Data...")
    df_train = load_data(split="train", load_cached_data=load_cached_data, nrows=nrows)

    print("Loading Validation Data...")
    df_val = load_data(split="val", load_cached_data=load_cached_data, nrows=nrows)

    # 3. Feature Extraction
    print("Extracting Features (Train)...")
    # This fits the vectorizer and saves it to Config.VECTORIZER_PATH
    X_train = extract_features(df_train, split="train")
    y_train = df_train[Config.TARGET_COL].values

    print("Extracting Features (Val)...")
    # This loads the vectorizer from Config.VECTORIZER_PATH and transforms
    X_val = extract_features(df_val, split="val")
    y_val = df_val[Config.TARGET_COL].values

    print(f"Train Features Shape: {X_train.shape}")
    print(f"Val Features Shape: {X_val.shape}")

    # 4. Model Training
    print("Initializing and Training Model...")
    model = ScoreRegressor()

    # The train method handles the grid search for alpha using the validation set
    model.train(X_train, y_train, X_val, y_val)

    # 5. Save Model
    print(f"Saving model to {Config.MODEL_PATH}...")
    model.save(Config.MODEL_PATH)

    # 6. Final Evaluation
    print("Performing Final Validation Evaluation...")
    # Generate integer predictions using optimized thresholds
    y_pred_val_int = model.predict_int(X_val)

    # Compute metric
    final_qwk = compute_qwk(y_val, y_pred_val_int)

    print(f"Final Validation QWK (Optimized Thresholds): {final_qwk}")
    print("--- Training Pipeline Completed ---")
