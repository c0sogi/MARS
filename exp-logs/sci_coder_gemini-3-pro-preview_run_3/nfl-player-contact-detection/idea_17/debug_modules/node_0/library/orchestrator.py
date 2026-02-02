import pandas as pd
import numpy as np
import os
import gc
import json
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataFactory
from library.feature_engine import FeatureEngine
from library.model_trainer import StreamTrainer, generate_submission


class Pipeline:
    """
    Orchestrates the Hybrid Coordinate Dual-Stream GBDT pipeline.
    Manages data loading, feature engineering, training, and inference.
    """

    def __init__(self):
        self.data_factory = DataFactory()
        self.feature_engine = FeatureEngine()
        self.thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")

    def run_training(self, debug=False, debug_size=1000):
        """
        Executes the training pipeline.
        1. Loads Train/Val data.
        2. Splits into Stream A (Interaction) and Stream B (Impact).
        3. Generates features.
        4. Trains models and optimizes thresholds.
        5. Saves models and thresholds.
        """
        seed_everything(Config.SEED)
        print("=== Starting Training Pipeline ===")

        # ---------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------
        # Train Data
        meta_train, track_train, helmets_train = self.data_factory.load_dataset(
            mode="train", load_cached_data=True, debug=debug, debug_size=debug_size
        )
        # Validation Data
        meta_val, track_val, helmets_val = self.data_factory.load_dataset(
            mode="validation", load_cached_data=True, debug=debug, debug_size=debug_size
        )

        # ---------------------------------------------------------
        # 2. Split Streams
        # ---------------------------------------------------------
        print(
            "Splitting metadata into Stream A (Player-Player) and Stream B (Player-Ground)..."
        )
        meta_train_a, meta_train_b = self.data_factory.split_contact_ids(meta_train)
        meta_val_a, meta_val_b = self.data_factory.split_contact_ids(meta_val)

        # ---------------------------------------------------------
        # 3. Feature Engineering
        # ---------------------------------------------------------
        # Stream A
        df_train_a = self.feature_engine.build_stream_a(
            meta_train_a,
            track_train,
            helmets_train,
            mode="train",
            load_cached_data=True,
        )
        df_val_a = self.feature_engine.build_stream_a(
            meta_val_a, track_val, helmets_val, mode="validation", load_cached_data=True
        )

        # Stream B
        df_train_b = self.feature_engine.build_stream_b(
            meta_train_b,
            track_train,
            helmets_train,
            mode="train",
            load_cached_data=True,
        )
        df_val_b = self.feature_engine.build_stream_b(
            meta_val_b, track_val, helmets_val, mode="validation", load_cached_data=True
        )

        # Free memory of raw data
        del meta_train, track_train, helmets_train, meta_val, track_val, helmets_val
        del meta_train_a, meta_train_b, meta_val_a, meta_val_b
        gc.collect()

        # ---------------------------------------------------------
        # 4. Training
        # ---------------------------------------------------------
        thresholds = {}

        # --- Train Stream A ---
        trainer_a = StreamTrainer(stream_name="StreamA")
        trainer_a.train(df_train_a, df_val_a)
        trainer_a.save_model("model_stream_a.json")
        thresholds["StreamA"] = float(trainer_a.best_threshold)

        # Free memory
        del df_train_a, df_val_a, trainer_a
        gc.collect()

        # --- Train Stream B ---
        trainer_b = StreamTrainer(stream_name="StreamB")
        trainer_b.train(df_train_b, df_val_b)
        trainer_b.save_model("model_stream_b.json")
        thresholds["StreamB"] = float(trainer_b.best_threshold)

        # Free memory
        del df_train_b, df_val_b, trainer_b
        gc.collect()

        # ---------------------------------------------------------
        # 5. Save Thresholds
        # ---------------------------------------------------------
        with open(self.thresholds_path, "w") as f:
            json.dump(thresholds, f)
        print(f"Thresholds saved to {self.thresholds_path}: {thresholds}")

        print("=== Training Pipeline Completed ===")

    def run_inference(self, debug=False, debug_size=1000):
        """
        Executes the inference pipeline.
        1. Loads Test data.
        2. Splits into Stream A and Stream B.
        3. Generates features.
        4. Loads models and thresholds.
        5. Generates predictions and submission file.
        """
        seed_everything(Config.SEED)
        print("=== Starting Inference Pipeline ===")

        # ---------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------
        meta_test, track_test, helmets_test = self.data_factory.load_dataset(
            mode="test", load_cached_data=True, debug=debug, debug_size=debug_size
        )

        # ---------------------------------------------------------
        # 2. Split Streams
        # ---------------------------------------------------------
        meta_test_a, meta_test_b = self.data_factory.split_contact_ids(meta_test)

        # ---------------------------------------------------------
        # 3. Feature Engineering
        # ---------------------------------------------------------
        # Stream A
        df_test_a = self.feature_engine.build_stream_a(
            meta_test_a, track_test, helmets_test, mode="test", load_cached_data=True
        )

        # Stream B
        df_test_b = self.feature_engine.build_stream_b(
            meta_test_b, track_test, helmets_test, mode="test", load_cached_data=True
        )

        # Free memory
        del meta_test, track_test, helmets_test, meta_test_a, meta_test_b
        gc.collect()

        # ---------------------------------------------------------
        # 4. Load Thresholds
        # ---------------------------------------------------------
        if not os.path.exists(self.thresholds_path):
            print("Warning: Thresholds file not found. Using default 0.5.")
            thresholds = {"StreamA": 0.5, "StreamB": 0.5}
        else:
            with open(self.thresholds_path, "r") as f:
                thresholds = json.load(f)

        print(f"Loaded thresholds: {thresholds}")

        # ---------------------------------------------------------
        # 5. Prediction
        # ---------------------------------------------------------

        # --- Predict Stream A ---
        trainer_a = StreamTrainer(stream_name="StreamA")
        # Initialize feature columns based on test data (excluding metadata)
        trainer_a.feature_cols = trainer_a._get_feature_cols(df_test_a)
        trainer_a.load_model("model_stream_a.json")
        trainer_a.best_threshold = thresholds.get("StreamA", 0.5)

        preds_a = trainer_a.predict(df_test_a)
        del df_test_a, trainer_a
        gc.collect()

        # --- Predict Stream B ---
        trainer_b = StreamTrainer(stream_name="StreamB")
        trainer_b.feature_cols = trainer_b._get_feature_cols(df_test_b)
        trainer_b.load_model("model_stream_b.json")
        trainer_b.best_threshold = thresholds.get("StreamB", 0.5)

        preds_b = trainer_b.predict(df_test_b)
        del df_test_b, trainer_b
        gc.collect()

        # ---------------------------------------------------------
        # 6. Submission
        # ---------------------------------------------------------
        generate_submission(preds_a, preds_b, Config.SUBMISSION_PATH)
        print("=== Inference Pipeline Completed ===")
