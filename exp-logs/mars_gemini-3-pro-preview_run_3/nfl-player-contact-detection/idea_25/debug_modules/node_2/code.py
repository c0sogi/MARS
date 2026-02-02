import pandas as pd
import numpy as np
import os
import sys

# Import library components
from library.utils import set_seed, calc_mcc
from library.data_loader import load_metadata, load_tracking, load_helmets
from library.feature_engineering import FeatureEngineer
from library.model_wrapper import ContactXGB
from library.config import WORKING_DIR


def main():
    print("=== Starting Demo Pipeline ===")

    # 1. Setup
    # Ensure reproducibility
    set_seed(42)
    print("[1/8] Setup complete.")

    # 2. Data Loading
    # We load a small subset of the training metadata to keep the demo fast.
    print("[2/8] Loading Data...")

    # Load first 5000 rows of metadata to find a few unique game plays
    df_meta_raw = load_metadata(mode="train", limit=5000)

    # Select the first 2 unique game_plays
    unique_games = df_meta_raw["game_play"].unique()[:2]
    print(f"      Selected Game Plays: {unique_games}")

    # Filter metadata to these plays
    df_meta_subset = df_meta_raw[df_meta_raw["game_play"].isin(unique_games)].copy()
    print(f"      Subset Labels Shape: {df_meta_subset.shape}")

    # Load Tracking and Helmet data specifically for these plays
    # This demonstrates the efficient loading capability of the data_loader
    df_tracking = load_tracking(mode="train", game_plays=unique_games)
    df_helmets = load_helmets(mode="train", game_plays=unique_games)

    # Validation assertions
    assert not df_tracking.empty, "Tracking data should not be empty."
    assert not df_helmets.empty, "Helmets data should not be empty."
    assert df_tracking["game_play"].nunique() == len(
        unique_games
    ), "Tracking data missing games."
    print("      Data loaded successfully.")

    # 3. Feature Engineering
    print("[3/8] Generating Features...")
    fe = FeatureEngineer()

    # Generate features for both streams
    # We use a custom mode 'demo' to create unique cache files for this run if needed
    # load_cached_data=False ensures we actually run the logic for this demo
    feature_data = fe.generate_features(
        df_labels=df_meta_subset,
        df_tracking=df_tracking,
        df_helmets=df_helmets,
        mode="demo",
        load_cached_data=False,
    )

    # Validate output structure
    assert "stream_a" in feature_data
    assert "stream_b" in feature_data
    assert "X" in feature_data["stream_a"]
    assert len(feature_data["stream_a"]["y"]) == len(feature_data["stream_a"]["X"])

    print(
        f"      Stream A (Player-Player) Samples: {len(feature_data['stream_a']['y'])}"
    )
    print(
        f"      Stream B (Player-Ground) Samples: {len(feature_data['stream_b']['y'])}"
    )

    # 4. Preparing Train/Validation Split
    # Since we are doing a demo, we manually split the generated features
    print("[4/8] Splitting Data...")

    def split_stream_data(stream_dict, split_ratio=0.8):
        X = stream_dict["X"]
        y = stream_dict["y"]
        ids = stream_dict["ids"]

        n = len(y)
        split_idx = int(n * split_ratio)

        # Handle DataFrame vs Numpy array for X
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        else:
            X_train, X_val = X[:split_idx], X[split_idx:]

        train_data = {"X": X_train, "y": y[:split_idx], "ids": ids[:split_idx]}
        val_data = {"X": X_val, "y": y[split_idx:], "ids": ids[split_idx:]}

        return train_data, val_data

    train_a, val_a = split_stream_data(feature_data["stream_a"])
    train_b, val_b = split_stream_data(feature_data["stream_b"])

    train_dict = {"stream_a": train_a, "stream_b": train_b}
    val_dict = {"stream_a": val_a, "stream_b": val_b}
    print("      Split complete.")

    # 5. Model Training
    print("[5/8] Training Models...")
    model = ContactXGB()

    # OPTIMIZATION: Drastically reduce n_estimators for speed in this demo
    model.model_a.n_estimators = 10
    model.model_b.n_estimators = 10

    # Fit the model
    model.fit(train_dict, val_dict)

    assert model.models_ready, "Models should be marked ready after fitting."
    print("      Training complete.")

    # 6. Threshold Optimization
    print("[6/8] Optimizing Thresholds...")
    # This adjusts the decision boundary based on validation MCC
    model.optimize_thresholds(val_dict)
    print(f"      Threshold A: {model.threshold_a:.4f}")
    print(f"      Threshold B: {model.threshold_b:.4f}")

    # 7. Prediction & Evaluation
    print("[7/8] Predicting and Evaluating...")

    # Predict on validation set
    preds_df = model.predict(val_dict)

    assert "contact_id" in preds_df.columns
    assert "contact" in preds_df.columns

    # Construct Ground Truth DataFrame for evaluation
    gt_a = pd.DataFrame({"contact_id": val_a["ids"], "y_true": val_a["y"]})
    gt_b = pd.DataFrame({"contact_id": val_b["ids"], "y_true": val_b["y"]})
    gt_df = pd.concat([gt_a, gt_b], ignore_index=True)

    # Merge Predictions with Ground Truth
    eval_df = pd.merge(preds_df, gt_df, on="contact_id", how="inner")

    if not eval_df.empty:
        mcc_score = calc_mcc(eval_df["y_true"], eval_df["contact"])
        print(f"      Validation MCC Score: {mcc_score:.4f}")
    else:
        print("      Warning: No overlapping IDs found for evaluation.")

    # 8. Model Persistence
    print("[8/8] Testing Save/Load...")
    save_path = "demo_contact_xgb.joblib"
    model.save(save_path)

    # Load into a fresh instance
    loaded_model = ContactXGB()
    loaded_model.load(save_path)

    # Verify state
    assert loaded_model.models_ready
    assert loaded_model.threshold_a == model.threshold_a
    assert loaded_model.threshold_b == model.threshold_b
    print("      Model state restored successfully.")

    print("=== Demo Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
