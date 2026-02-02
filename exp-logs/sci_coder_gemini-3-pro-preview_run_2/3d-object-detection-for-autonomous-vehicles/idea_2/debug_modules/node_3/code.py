import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import random
import torch
import joblib


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import Config
from library.utils import (
    CalibrationRegistry,
    box_iou_3d,
    make_transform_matrix,
    transform_points,
    convert_box_to_global,
)
from library.data_processing import (
    PointCloudProcessor,
    FeatureExtractor,
    create_training_dataset,
)
from library.model import ClusterClassifier
from library.train import Trainer
from library.inference import InferencePipeline, generate_submission

if __name__ == "__main__":
    print("=== Starting Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Use a separate working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config attributes directly
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set debug size to process only a few samples
    Config.DEBUG_SAMPLE_SIZE = 5

    # Reduce model complexity for instant training
    Config.XGB_CLF_PARAMS["n_estimators"] = 2
    Config.XGB_CLF_PARAMS["n_jobs"] = 1
    Config.XGB_REG_PARAMS["n_estimators"] = 2
    Config.XGB_REG_PARAMS["n_jobs"] = 1

    print(f"Working Directory set to: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Demonstrating Library Utils
    # ---------------------------------------------------------
    print("\n[2] Testing Library Utils...")

    # A. Calibration Registry
    # This requires reading the input JSONs.
    registry = CalibrationRegistry(Config.INPUT_DIR, Config.WORKING_DIR)
    # We can't easily guess a valid token without reading metadata, but we can check if the table loaded
    assert isinstance(
        registry.lookup_table, dict
    ), "CalibrationRegistry failed to load lookup table."
    print("CalibrationRegistry loaded successfully.")

    # B. 3D IoU Calculation
    # Box format: [x, y, z, w, l, h, yaw]
    box_a = np.array([0, 0, 0, 2, 4, 2, 0])
    box_b = np.array([0, 0, 0, 2, 4, 2, 0])  # Identical box
    iou_perfect = box_iou_3d(box_a, box_b)
    assert np.isclose(
        iou_perfect, 1.0
    ), f"Expected IoU 1.0 for identical boxes, got {iou_perfect}"

    box_c = np.array([10, 10, 10, 2, 4, 2, 0])  # Disjoint box
    iou_zero = box_iou_3d(box_a, box_c)
    assert np.isclose(
        iou_zero, 0.0
    ), f"Expected IoU 0.0 for disjoint boxes, got {iou_zero}"
    print("3D IoU calculation verified.")

    # C. Geometric Transforms
    # Translation of [1, 0, 0]
    mat = make_transform_matrix(
        [1, 0, 0], [1, 0, 0, 0]
    )  # Identity rotation quaternion [w, x, y, z]
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    transformed = transform_points(points, mat)

    expected = np.array([[1, 0, 0], [2, 1, 1]], dtype=np.float32)
    assert np.allclose(transformed[:, :3], expected), "Point transformation failed."
    print("Geometric transforms verified.")

    # ---------------------------------------------------------
    # 3. Demonstrating Data Processing (Synthetic Data)
    # ---------------------------------------------------------
    print("\n[3] Testing Data Processing components...")

    processor = PointCloudProcessor()

    # Create synthetic point cloud
    # 1. Ground plane points (z ~ -2.0)
    ground_x = np.random.uniform(-10, 10, 100)
    ground_y = np.random.uniform(-10, 10, 100)
    ground_z = np.full(100, -2.0) + np.random.normal(0, 0.05, 100)
    ground_pts = np.column_stack([ground_x, ground_y, ground_z])

    # 2. Object points (cluster at 5, 5, 0)
    obj_x = np.random.normal(5, 0.5, 20)
    obj_y = np.random.normal(5, 0.5, 20)
    obj_z = np.random.normal(0, 0.5, 20)
    obj_pts = np.column_stack([obj_x, obj_y, obj_z])

    # Combine
    all_points = np.vstack([ground_pts, obj_pts]).astype(np.float32)

    # A. Preprocess (ROI)
    # Config ROI is large, so all these points should stay
    roi_points = processor.preprocess(all_points)
    assert len(roi_points) == len(
        all_points
    ), "ROI filtering removed points unexpectedly."

    # B. Ground Removal
    # This should remove the ground plane points
    points_no_ground = processor.remove_ground(roi_points)
    # We expect roughly 20 points left (the object)
    # RANSAC isn't deterministic with small samples/iterations sometimes, but with perfect plane it should work
    assert (
        len(points_no_ground) < 50
    ), f"Ground removal failed, left {len(points_no_ground)} points."

    # C. Clustering
    clusters = processor.cluster_points(points_no_ground)
    # Should find at least one cluster (the object)
    assert len(clusters) >= 1, "DBSCAN failed to cluster object points."

    # D. Feature Extraction
    extractor = FeatureExtractor()
    feats = extractor.extract(clusters[0])

    required_keys = [
        "point_count",
        "x_mean",
        "cluster_volume",
        "eigenvalues" if "eigenvalues" in feats else "eig_1",
    ]
    for k in ["point_count", "x_mean", "cluster_volume"]:
        assert k in feats, f"Feature extractor missing key: {k}"

    print(
        f"Data Processing verified. Extracted features for synthetic cluster: {feats['point_count']} points."
    )

    # ---------------------------------------------------------
    # 4. Generating Training Dataset (Real Data Subset)
    # ---------------------------------------------------------
    print("\n[4] Generating Training Dataset (Subset)...")

    train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")

    # This function uses Config.DEBUG_SAMPLE_SIZE internally to limit rows
    df_train = create_training_dataset(train_meta_path, load_cached_data=False)

    if df_train.empty:
        print(
            "Warning: No clusters found in the debug subset. This might happen if samples are empty."
        )
    else:
        print(f"Generated training dataset with {len(df_train)} rows.")
        assert (
            "target_class" in df_train.columns
        ), "Dataset missing target_class column."
        assert (
            "is_background" in df_train.columns
        ), "Dataset missing is_background column."

    # ---------------------------------------------------------
    # 5. Training the Model
    # ---------------------------------------------------------
    print("\n[5] Running Training Pipeline...")

    trainer = Trainer()

    # We force generation of validation data as well (it will use the debug size)
    # Note: Trainer.train() calls create_training_dataset internally.
    # Since we already generated train data above and it cached it to Config.WORKING_DIR/train_features.parquet,
    # Trainer will pick it up if load_cached_data=True.

    try:
        trainer.train(load_cached_data=True)
    except Exception as e:
        # If training fails (e.g. empty dataset due to random sampling of empty scenes), we fail explicitly
        raise RuntimeError(f"Training pipeline failed: {e}")

    model_path = os.path.join(Config.WORKING_DIR, "model.joblib")
    assert os.path.exists(model_path), "Model file was not created."
    print("Model training complete and saved.")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Running Inference Pipeline...")

    # Initialize pipeline with the model we just trained
    pipeline = InferencePipeline(model_path=model_path)

    # Run generation on a debug subset of test data
    # We pass debug_size explicitly here
    pipeline.generate_submission(debug_size=5)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    assert "Id" in df_sub.columns, "Submission missing Id column."
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing PredictionString column."
    assert len(df_sub) == 5, f"Expected 5 predictions, got {len(df_sub)}"

    print(f"Submission generated at {submission_path}")
    print("First row of submission:")
    print(df_sub.iloc[0])

    print("\n=== Demonstration Complete Successfully ===")
