import os
import sys
import numpy as np
import pandas as pd
import logging
import shutil

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.feature_extraction import DualStreamExtractor
from library.densification import ManifoldDensifier
from library.cross_validation import EnsembleTrainer
from library.inference import InferenceEngine


# Define a custom configuration for the demo run to ensure speed
class DemoConfig(Config):
    def __init__(self):
        super().__init__(debug=True, limit_dataset=12)
        # Override working directory to separate demo outputs
        self.WORKING_DIR = "./working/demo_run"
        self.MODELS_DIR = os.path.join(self.WORKING_DIR, "models")

        # Override training parameters for speed
        self.N_FOLDS = 2

        # Ensure directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)


def main():
    # 1. Setup and Configuration
    print(">>> Step 1: Initialization")
    seed_everything(42)
    config = DemoConfig()
    setup_logging(os.path.join(config.WORKING_DIR, "execution.log"))

    # Verify paths
    assert os.path.exists(config.TRAIN_METADATA_PATH), "Train metadata not found"
    assert os.path.exists(config.TEST_METADATA_PATH), "Test metadata not found"
    print("Initialization complete.")

    # 2. Feature Extraction (Train Set)
    print("\n>>> Step 2: Feature Extraction (Train)")
    extractor = DualStreamExtractor(config)

    # Process training data (limited to 12 samples by config)
    # This extracts 12 rotations per image for DINOv2 and ConvNeXt
    train_data = extractor.process_dataset(
        config.TRAIN_METADATA_PATH, dataset_name="demo_train", load_cached_data=False
    )

    # Verification of Extracted Data
    n_samples = len(train_data["ids"])
    print(f"Extracted {n_samples} samples.")

    # Check shapes: (N, 12, D) for visual features
    assert train_data["dino"].ndim == 3, "DINO features should be 3D (N, Views, Dim)"
    assert train_data["dino"].shape[1] == 12, "DINO features should have 12 views"
    assert train_data["conv"].ndim == 3, "ConvNeXt features should be 3D"
    assert train_data["conv"].shape[1] == 12, "ConvNeXt features should have 12 views"
    assert (
        train_data["tab"].shape[0] == n_samples
    ), "Tabular features row count mismatch"
    assert len(train_data["labels"]) == n_samples, "Labels count mismatch"

    # Hack: Overwrite labels to ensure StratifiedKFold works with N=2 on this tiny dataset
    # We create a binary classification scenario for the demo
    print("Synthesizing labels for robust StratifiedKFold on tiny dataset...")
    train_data["labels"] = np.array(
        ["Class_A" if i % 2 == 0 else "Class_B" for i in range(n_samples)]
    )

    # 3. Manifold Densification
    print("\n>>> Step 3: Manifold Densification")
    densifier = ManifoldDensifier(config)

    # Densify the training data (Convex Hull generation)
    # This expands N samples -> 6*N samples (centroids)
    dense_train_data = densifier.densify_dataset(
        train_data, dataset_name="demo_train", load_cached_data=False
    )

    # Verification of Densification
    n_dense = len(dense_train_data["ids"])
    print(f"Densified samples: {n_dense}")

    assert n_dense == n_samples * 6, "Densification should result in 6x samples"
    assert (
        dense_train_data["dino"].ndim == 2
    ), "Densified visual features should be flattened (2D)"
    assert (
        dense_train_data["ids"][0] == dense_train_data["ids"][1]
    ), "IDs should be repeated for centroids"

    # 4. Ensemble Training
    print("\n>>> Step 4: Ensemble Training")
    trainer = EnsembleTrainer(config)

    # Train the pipeline (PCA -> QuantileTransformer -> LDA)
    # Note: We pass the raw (12-view) data to train_loop, it handles densification internally per fold
    avg_score = trainer.train_loop(train_data)

    print(f"Training complete. Average Score: {avg_score:.4f}")

    # Verify Model Artifacts
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "models", "classes.pkl")
    ), "Classes file missing"
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "models", "pipeline_fold_0.pkl")
    ), "Fold 0 model missing"
    assert isinstance(avg_score, float), "Score should be a float"

    # 5. Inference
    print("\n>>> Step 5: Inference")
    inference_engine = InferenceEngine(config)

    # Run prediction on the test set (limited to 12 samples)
    # This runs extraction, densification, prediction, and aggregation
    predictions_df = inference_engine.predict_all(
        config.TEST_METADATA_PATH, dataset_name="demo_test"
    )

    print("Inference complete.")
    print(predictions_df.head())

    # Verification of Predictions
    assert "id" in predictions_df.columns, "Predictions DataFrame missing 'id' column"
    # We used synthetic binary labels (Class_A, Class_B), so we expect these columns
    assert "Class_A" in predictions_df.columns, "Missing predicted class column"
    assert len(predictions_df) == min(
        12, len(pd.read_csv(config.TEST_METADATA_PATH))
    ), "Prediction row count mismatch"

    # 6. Submission Generation
    print("\n>>> Step 6: Submission Generation")
    submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    inference_engine.generate_submission(config.TEST_METADATA_PATH, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created"
    print(f"Submission saved to {submission_path}")

    print("\n>>> Demo Execution Successful!")


if __name__ == "__main__":
    main()
