import os
import sys
import pandas as pd
import numpy as np
import torch

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, load_dataset, save_submission
from library.features import FeatureGenerator
from library.rf_model import MultiInteractionRF
from library.mlp_trainer import MLPTrainer


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print(">>> 1. Configuring environment for fast demonstration...")

    # Override Config for speed and low resource usage
    Config.MLP_EPOCHS = 2
    Config.MLP_BATCH_SIZE = 4
    Config.RF_N_ESTIMATORS = 10
    Config.TFIDF_MAX_FEATURES = 50  # Reduce vocabulary size
    Config.TOP_K_SUBREDDITS = 5  # Reduce dimensionality of community features

    # Set seed for reproducibility
    set_seed(42)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print(">>> 2. Loading data subsets...")
    SAMPLE_SIZE = 20

    # Load small samples to verify pipeline flow
    df_train = load_dataset("train", sample_size=SAMPLE_SIZE)
    df_val = load_dataset("val", sample_size=SAMPLE_SIZE)
    df_test = load_dataset("test", sample_size=SAMPLE_SIZE)

    # Verify data loading
    assert len(df_train) == SAMPLE_SIZE
    assert len(df_val) == SAMPLE_SIZE
    assert len(df_test) == SAMPLE_SIZE
    assert Config.TARGET_COL in df_train.columns
    print(f"Loaded {SAMPLE_SIZE} samples for train, val, and test.")

    # --------------------------------------------------------------------------
    # 3. Feature Generation
    # --------------------------------------------------------------------------
    print(">>> 3. Generating features...")
    fg = FeatureGenerator()

    # Fit on training data (learns TF-IDF vocab, scalers, top-k subreddits)
    fg.fit(df_train)

    # Transform all splits
    # Returns: (mlp_features_dict, rf_features_array, labels)
    train_mlp, train_rf, train_y = fg.transform(df_train)
    val_mlp, val_rf, val_y = fg.transform(df_val)
    test_mlp, test_rf, test_y = fg.transform(df_test)

    # Verify Feature Dimensions
    # RF Features: TFIDF(50) + Meta(7) + TopK(5) + Interaction(4) + Sentiment(4) + Consistency(2) = 72
    expected_rf_dim = 50 + 7 + 5 + 4 + 4 + 2
    assert train_rf.shape == (
        SAMPLE_SIZE,
        expected_rf_dim,
    ), f"Expected RF dim {expected_rf_dim}, got {train_rf.shape[1]}"

    # MLP Features: Check dictionary keys
    expected_keys = [
        "title_emb",
        "body_emb",
        "metadata",
        "top_k",
        "history_centroid",
        "consistency",
        "sentiment",
    ]
    for key in expected_keys:
        assert key in train_mlp, f"Missing key {key} in MLP features"
        assert len(train_mlp[key]) == SAMPLE_SIZE

    print("Feature generation verified.")

    # --------------------------------------------------------------------------
    # 4. Random Forest Model (Stream A)
    # --------------------------------------------------------------------------
    print(">>> 4. Training Random Forest (Stream A)...")
    rf_model = MultiInteractionRF()

    # Train
    rf_model.train(train_rf, train_y, val_rf, val_y)

    # Predict
    rf_preds = rf_model.predict(test_rf)

    # Verify Predictions
    assert len(rf_preds) == SAMPLE_SIZE
    assert (rf_preds >= 0).all() and (
        rf_preds <= 1
    ).all(), "RF predictions out of bounds"
    print("Random Forest pipeline verified.")

    # --------------------------------------------------------------------------
    # 5. MLP Model (Stream B)
    # --------------------------------------------------------------------------
    print(">>> 5. Training MLP (Stream B)...")
    mlp_trainer = MLPTrainer()

    # Train
    mlp_trainer.train((train_mlp, train_y), (val_mlp, val_y))

    # Predict
    mlp_preds = mlp_trainer.predict(test_mlp)

    # Verify Predictions
    assert len(mlp_preds) == SAMPLE_SIZE
    assert (mlp_preds >= 0).all() and (
        mlp_preds <= 1
    ).all(), "MLP predictions out of bounds"
    print("MLP pipeline verified.")

    # --------------------------------------------------------------------------
    # 6. Ensemble & Submission
    # --------------------------------------------------------------------------
    print(">>> 6. Creating Ensemble and Submission...")

    # Simple Average Ensemble
    final_preds = (rf_preds * 0.5) + (mlp_preds * 0.5)

    # Format for submission
    submission_df = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: final_preds}
    )

    # Save submission
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(submission_df, filename=demo_submission_path)

    # Verify file output
    assert os.path.exists(demo_submission_path), "Submission file not created"

    # Verify content
    loaded_sub = pd.read_csv(demo_submission_path)
    assert len(loaded_sub) == SAMPLE_SIZE
    assert list(loaded_sub.columns) == [Config.ID_COL, Config.TARGET_COL]

    print(f"Submission saved to {demo_submission_path}")
    print(">>> Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
