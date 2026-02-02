import os
import json
import numpy as np
import pandas as pd
import tqdm

from library.config import Config
from library.utils import load_point_cloud, CalibrationRegistry, convert_box_to_global
from library.data_processing import PointCloudProcessor, FeatureExtractor
from library.model import ClusterClassifier


class InferencePipeline:
    """
    Manages the inference process for the 3D Object Detection task.
    Loads the trained model, processes test samples, and generates the submission file.
    """

    def __init__(self, model_path=None):
        self.test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.csv")
        self.submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        # Default model path if not provided
        if model_path is None:
            self.model_path = os.path.join(Config.WORKING_DIR, "model.joblib")
        else:
            self.model_path = model_path

        # Initialize Components
        print("Initializing Inference Components...")
        self.registry = CalibrationRegistry(Config.INPUT_DIR, Config.WORKING_DIR)
        self.processor = PointCloudProcessor()
        self.extractor = FeatureExtractor()

        # Load Model
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = ClusterClassifier.load(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _format_prediction_string(self, predictions):
        """
        Converts a list of global prediction dictionaries into the submission string format.
        Format: confidence center_x center_y center_z width length height yaw class_name
        """
        if not predictions:
            return ""

        parts = []
        # Sort by confidence descending
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        for p in predictions:
            # Rounding to reasonable precision to save space, though not strictly required
            # Format: conf x y z w l h yaw class
            s = (
                f"{p['confidence']:.4f} {p['center_x']:.4f} {p['center_y']:.4f} {p['center_z']:.4f} "
                f"{p['width']:.4f} {p['length']:.4f} {p['height']:.4f} {p['yaw']:.4f} {p['class_name']}"
            )
            parts.append(s)

        return " ".join(parts)

    def process_sample(self, sample_token, file_paths):
        """
        End-to-end processing of a single test sample.
        Returns a list of prediction dictionaries in Global Frame.
        """
        # 1. Identify LiDAR File
        lidar_path = None
        lidar_channel = None

        # file_paths is a dict
        for ch, p in file_paths.items():
            if "LIDAR" in ch or p.endswith(".bin"):
                lidar_path = os.path.join(Config.INPUT_DIR, p)
                lidar_channel = ch
                break

        if not lidar_path or not os.path.exists(lidar_path):
            return []

        # 2. Get Calibration Transforms
        # We need Sensor -> Ego -> Global
        mat_se, mat_eg = self.registry.get_transform(sample_token, lidar_channel)
        if mat_se is None:
            return []

        # 3. Load and Process Point Cloud
        points = load_point_cloud(lidar_path)
        points = self.processor.preprocess(points)
        points_no_ground = self.processor.remove_ground(points)
        clusters = self.processor.cluster_points(points_no_ground)

        if not clusters:
            return []

        # 4. Extract Features
        feature_rows = []
        for cluster in clusters:
            feats = self.extractor.extract(cluster)
            if feats is not None:
                feats["sample_token"] = sample_token
                feature_rows.append(feats)

        if not feature_rows:
            return []

        df_features = pd.DataFrame(feature_rows)

        # 5. Predict (Returns predictions in Sensor Frame)
        # Model.predict returns:
        # [{'sample_token':..., 'confidence':..., 'center_x':..., ...}, ...]
        sensor_preds = self.model.predict(df_features)

        if not sensor_preds:
            return []

        # 6. Transform to Global Frame
        global_preds = []
        for pred in sensor_preds:
            # Construct box array [x, y, z, w, l, h, yaw]
            box_s = np.array(
                [
                    pred["center_x"],
                    pred["center_y"],
                    pred["center_z"],
                    pred["width"],
                    pred["length"],
                    pred["height"],
                    pred["yaw"],
                ]
            )

            # Convert
            box_g = convert_box_to_global(box_s, mat_se, mat_eg)

            # Re-package
            global_preds.append(
                {
                    "confidence": pred["confidence"],
                    "center_x": box_g[0],
                    "center_y": box_g[1],
                    "center_z": box_g[2],
                    "width": box_g[3],
                    "length": box_g[4],
                    "height": box_g[5],
                    "yaw": box_g[6],
                    "class_name": pred["class_name"],
                }
            )

        return global_preds

    def generate_submission(self, debug_size=None):
        """
        Main loop to generate the submission file.
        """
        print("Loading test metadata...")
        if not os.path.exists(self.test_meta_path):
            raise FileNotFoundError(f"Test metadata not found at {self.test_meta_path}")

        df_test = pd.read_csv(self.test_meta_path)

        # Parse JSON file_paths
        df_test["file_paths"] = df_test["file_paths"].apply(json.loads)

        if debug_size is not None:
            print(f"Debug mode: Processing first {debug_size} samples.")
            df_test = df_test.iloc[:debug_size]

        print(f"Generating predictions for {len(df_test)} samples...")
        print(f"Saving to {self.submission_path}")

        # Open file and write header
        with open(self.submission_path, "w") as f:
            f.write("Id,PredictionString\n")

            # Iterate samples
            # Using simple loop to avoid external dependency issues if tqdm not available,
            # but usually tqdm is standard.
            count = 0
            for idx, row in df_test.iterrows():
                sample_token = row["token"]
                file_paths = row["file_paths"]

                try:
                    preds = self.process_sample(sample_token, file_paths)
                    pred_str = self._format_prediction_string(preds)
                except Exception as e:
                    # Fallback to empty prediction on error to ensure submission integrity
                    # print(f"Error processing {sample_token}: {e}")
                    pred_str = ""

                f.write(f"{sample_token},{pred_str}\n")

                count += 1
                if count % 500 == 0:
                    print(f"Processed {count}/{len(df_test)} samples...")

        print("Submission generation complete.")


def generate_submission(debug_size=None):
    """
    Wrapper function to execute the pipeline.
    """
    pipeline = InferencePipeline()
    pipeline.generate_submission(debug_size=debug_size)
