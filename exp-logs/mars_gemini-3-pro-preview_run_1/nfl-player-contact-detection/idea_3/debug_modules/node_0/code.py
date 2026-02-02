import os
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.data_pipeline import DataPipeline
from library.model_interaction import InteractionGBM
from library.model_impact import ImpactTrainer
from library.utils import seed_everything


def create_mini_datasets():
    """
    Creates a small subset of the training data in the ./working directory
    to allow for fast demonstration and verification of the pipeline.
    """
    print("Creating mini datasets for fast verification...")

    # Load original metadata and tracking (just enough to get a few plays)
    # We read the first 100k rows to ensure we capture full plays
    meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH, nrows=10000)
    track_df = pd.read_csv(Config.TRAIN_TRACKING_PATH, nrows=50000)

    # Select a few unique plays to maintain integrity
    unique_plays = meta_df["game_play"].unique()
    if len(unique_plays) < 3:
        raise ValueError("Not enough plays in the head of metadata for split.")

    train_plays = unique_plays[:2]
    val_plays = unique_plays[2:3]

    # Filter Metadata
    mini_train_meta = meta_df[meta_df["game_play"].isin(train_plays)].copy()
    mini_val_meta = meta_df[meta_df["game_play"].isin(val_plays)].copy()

    # Filter Tracking
    # We need tracking data for the plays selected
    relevant_plays = np.concatenate([train_plays, val_plays])
    mini_tracking = track_df[track_df["game_play"].isin(relevant_plays)].copy()

    # Ensure we have tracking data for these plays.
    # If the head of tracking csv doesn't match head of metadata plays, we might need to be careful.
    # In this dataset, they are usually sorted, but let's verify.
    if mini_tracking.empty:
        # Fallback: Create mock tracking data if the head didn't align
        print(
            "Warning: Tracking data head did not align with metadata plays. Creating mock tracking."
        )
        mini_tracking = pd.DataFrame(columns=track_df.columns)
        for play in relevant_plays:
            # Create dummy steps 0 to 50
            steps = np.arange(50)
            # Create dummy players
            players = mini_train_meta[mini_train_meta["game_play"] == play][
                "nfl_player_id_1"
            ].unique()
            if len(players) == 0:
                players = ["12345", "67890"]

            for pid in players:
                rows = pd.DataFrame(
                    {
                        "game_play": play,
                        "game_key": play.split("_")[0],
                        "play_id": play.split("_")[1],
                        "nfl_player_id": pid,
                        "datetime": "2020-01-01",
                        "step": steps,
                        "position": "QB",
                        "team": "Home",
                        "jersey_number": 1,
                        "x_position": np.random.uniform(0, 100, size=len(steps)),
                        "y_position": np.random.uniform(0, 50, size=len(steps)),
                        "speed": np.random.uniform(0, 10, size=len(steps)),
                        "distance": np.random.uniform(0, 1, size=len(steps)),
                        "orientation": np.random.uniform(0, 360, size=len(steps)),
                        "direction": np.random.uniform(0, 360, size=len(steps)),
                        "acceleration": np.random.uniform(0, 5, size=len(steps)),
                        "sa": np.random.uniform(-2, 2, size=len(steps)),
                    }
                )
                mini_tracking = pd.concat([mini_tracking, rows], ignore_index=True)

    # Save Mini Files
    mini_train_meta_path = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    mini_tracking_path = os.path.join(Config.WORKING_DIR, "mini_train_tracking.csv")

    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    mini_tracking.to_csv(mini_tracking_path, index=False)

    print(f"Mini Train Meta: {len(mini_train_meta)} rows")
    print(f"Mini Val Meta: {len(mini_val_meta)} rows")
    print(f"Mini Tracking: {len(mini_tracking)} rows")

    return mini_train_meta_path, mini_val_meta_path, mini_tracking_path


def configure_environment(train_meta_path, val_meta_path, tracking_path):
    """
    Overrides the global Config to use the mini datasets and faster training parameters.
    """
    # Override Paths
    Config.TRAIN_METADATA_PATH = train_meta_path
    Config.VAL_METADATA_PATH = val_meta_path
    Config.TRAIN_TRACKING_PATH = tracking_path

    # Override Model Hyperparameters for Speed
    Config.LGBM_TRAIN_PARAMS["num_boost_round"] = 10
    Config.LGBM_TRAIN_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_TRAIN_PARAMS["verbose_eval"] = -1

    Config.CNN_TRAIN_PARAMS["epochs"] = 2
    Config.CNN_TRAIN_PARAMS["batch_size"] = 16
    Config.CNN_TRAIN_PARAMS["num_workers"] = (
        0  # Avoid multiprocessing overhead for tiny data
    )


def verify_data_pipeline():
    print("\n=== Verifying Data Pipeline ===")
    pipeline = DataPipeline(Config)

    # Force re-creation of features (load_cached=False) to test logic
    datasets = pipeline.prepare_datasets(load_cached=False)

    # Verify Stream A (Interaction)
    stream_a = datasets["stream_a"]
    train_df_a = stream_a["train"]
    val_df_a = stream_a["val"]
    feats_a = stream_a["features"]

    print(f"Stream A Train Shape: {train_df_a.shape}")
    print(f"Stream A Features: {len(feats_a)} columns")

    assert not train_df_a.empty, "Stream A training dataframe is empty."
    assert not val_df_a.empty, "Stream A validation dataframe is empty."
    assert "contact" in train_df_a.columns, "Target column missing in Stream A."
    assert len(feats_a) > 0, "No features detected for Stream A."

    # Verify Stream B (Impact)
    stream_b = datasets["stream_b"]
    X_train_b, y_train_b = stream_b["train"]
    X_val_b, y_val_b = stream_b["val"]

    print(f"Stream B Train X Shape: {X_train_b.shape}")

    assert len(X_train_b) > 0, "Stream B training data is empty."
    assert (
        X_train_b.ndim == 3
    ), f"Stream B input should be 3D (N, C, L), got {X_train_b.shape}"
    assert X_train_b.shape[1] == len(
        Config.STREAM_B_FEATURES
    ), "Incorrect channel count in Stream B."
    assert (
        X_train_b.shape[2] == Config.WINDOW_SIZE * 2 + 1
    ), "Incorrect sequence length in Stream B."

    return datasets


def verify_interaction_model(datasets):
    print("\n=== Verifying Interaction Model (LightGBM) ===")

    train_df = datasets["stream_a"]["train"]
    val_df = datasets["stream_a"]["val"]
    features = datasets["stream_a"]["features"]

    # Ensure there are both classes in the mini-set for training to work
    if train_df["contact"].nunique() < 2:
        print("Injecting dummy positive sample for training stability...")
        dummy = train_df.iloc[0].copy()
        dummy["contact"] = 1 - dummy["contact"]
        train_df = pd.concat([train_df, pd.DataFrame([dummy])], ignore_index=True)

    model = InteractionGBM(Config)

    # Train
    model.train(train_df, val_df, features)

    # Predict
    preds = model.predict(val_df, features)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Predictions Mean: {preds.mean():.4f}")

    assert preds.shape[0] == len(val_df), "Prediction count mismatch."
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of probability range."
    assert os.path.exists(model.model_path), "Model file was not saved."


def verify_impact_model(datasets):
    print("\n=== Verifying Impact Model (1D-CNN) ===")

    X_train, y_train = datasets["stream_b"]["train"]
    X_val, y_val = datasets["stream_b"]["val"]

    # Ensure targets are float for BCEWithLogits
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    trainer = ImpactTrainer(Config)

    # Train
    trainer.train(X_train, y_train, X_val, y_val)

    # Predict
    preds = trainer.predict(X_val)

    print(f"Predictions Shape: {preds.shape}")

    assert preds.shape[0] == len(X_val), "Prediction count mismatch."
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of probability range."
    assert os.path.exists(trainer.model_path), "Model file was not saved."


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Setup Mini Data
    train_meta, val_meta, tracking = create_mini_datasets()

    # 2. Configure Config to use Mini Data
    configure_environment(train_meta, val_meta, tracking)

    # 3. Verify Data Pipeline
    datasets = verify_data_pipeline()

    # 4. Verify Interaction Model
    verify_interaction_model(datasets)

    # 5. Verify Impact Model
    verify_impact_model(datasets)

    print("\nSUCCESS: All components verified successfully.")
