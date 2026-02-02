import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from library.config import Config
from library.utils import calc_iou_3d, get_transform_matrix
from library.cluster_proposal import GeometricProposalGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def parse_label_string(label_str):
    """
    Parses the space-delimited label string into a list of dictionaries.
    Format: center_x center_y center_z width length height yaw class_name
    """
    if pd.isna(label_str) or label_str == "":
        return []

    parts = str(label_str).strip().split()
    stride = 8
    # If the string is malformed or doesn't match stride, return empty
    if len(parts) % stride != 0:
        return []

    num_objects = len(parts) // stride
    objects = []
    for i in range(num_objects):
        offset = i * stride
        try:
            box = np.array(
                [float(parts[offset + j]) for j in range(7)], dtype=np.float32
            )
            class_name = parts[offset + 7]
            objects.append({"box": box, "class_name": class_name})
        except ValueError:
            continue
    return objects


def compute_eigen_features(points):
    """
    Computes eigenvalue-based shape descriptors from the covariance matrix of points.
    """
    # Default values for empty or small clusters
    defaults = {
        "eigen_1": 0.0,
        "eigen_2": 0.0,
        "eigen_3": 0.0,
        "linearity": 0.0,
        "planarity": 0.0,
        "sphericity": 0.0,
        "omnivariance": 0.0,
        "anisotropy": 0.0,
        "eigenentropy": 0.0,
        "sum_eigen": 0.0,
        "change_curvature": 0.0,
    }

    if len(points) < 4:
        return defaults

    try:
        # Covariance of XYZ coordinates
        cov = np.cov(points[:, :3], rowvar=False)
        # eigenvalues in ascending order
        eigenvalues = np.linalg.eigvalsh(cov)
    except Exception:
        return defaults

    # Sort descending: l1 >= l2 >= l3
    l3, l2, l1 = eigenvalues

    # Avoid division by zero
    l1 = max(l1, 1e-9)
    l2 = max(l2, 1e-9)
    l3 = max(l3, 1e-9)

    sum_eigen = l1 + l2 + l3

    # Shape factors
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    omnivariance = (l1 * l2 * l3) ** (1 / 3)
    anisotropy = (l1 - l3) / l1
    change_curvature = l3 / sum_eigen

    # Eigenentropy
    probs = eigenvalues / sum_eigen
    probs = np.maximum(probs, 1e-12)  # Numerical stability
    eigenentropy = -np.sum(probs * np.log(probs))

    return {
        "eigen_1": l1,
        "eigen_2": l2,
        "eigen_3": l3,
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "omnivariance": omnivariance,
        "anisotropy": anisotropy,
        "eigenentropy": eigenentropy,
        "sum_eigen": sum_eigen,
        "change_curvature": change_curvature,
    }


def extract_features_single(proposal, sample_token):
    """
    Extracts geometric and statistical features for a single proposal.
    """
    box = proposal["box"]
    points = proposal["points"]

    # Unpack box: x, y, z, w, l, h, yaw
    cx, cy, cz, w, l, h, yaw = box

    # 1. Eigen Features
    feats = compute_eigen_features(points)

    # 2. Spatial Features
    feats["center_x"] = cx
    feats["center_y"] = cy
    feats["center_z"] = cz
    feats["bbox_width"] = w
    feats["bbox_length"] = l
    feats["bbox_height"] = h
    feats["bbox_volume"] = w * l * h
    feats["bbox_yaw"] = yaw

    # 3. Statistical Features
    num_points = len(points)
    feats["num_points"] = num_points
    feats["point_density"] = num_points / max(feats["bbox_volume"], 1e-6)

    # Intensity stats (4th column)
    if points.shape[1] >= 4 and num_points > 0:
        intensities = points[:, 3]
        feats["intensity_min"] = np.min(intensities)
        feats["intensity_max"] = np.max(intensities)
        feats["intensity_mean"] = np.mean(intensities)
        feats["intensity_std"] = np.std(intensities)
    else:
        feats["intensity_min"] = 0.0
        feats["intensity_max"] = 0.0
        feats["intensity_mean"] = 0.0
        feats["intensity_std"] = 0.0

    # Add metadata
    feats["sample_token"] = sample_token

    return feats


def compute_residuals(prop_box, gt_box):
    """
    Computes regression targets (residuals) between proposal and ground truth.
    """
    px, py, pz, pw, pl, ph, pyaw = prop_box
    gx, gy, gz, gw, gl, gh, gyaw = gt_box

    dx = gx - px
    dy = gy - py
    dz = gz - pz

    # Log-scale dimensions
    dw = np.log(gw / max(pw, 1e-6))
    dl = np.log(gl / max(pl, 1e-6))
    dh = np.log(gh / max(ph, 1e-6))

    # Yaw difference (normalized to -pi, pi)
    dyaw = gyaw - pyaw
    while dyaw > np.pi:
        dyaw -= 2 * np.pi
    while dyaw < -np.pi:
        dyaw += 2 * np.pi

    return {"dx": dx, "dy": dy, "dz": dz, "dw": dw, "dl": dl, "dh": dh, "dyaw": dyaw}


def process_sample_wrapper(row, is_test=False):
    """
    Worker function to process a single sample.
    Designed to be pickle-able for joblib.
    """
    sample_token = row["sample_token"]
    lidar_path = row["lidar_path"]
    transform_matrix = row.get("transform_matrix")

    # Initialize generator locally
    generator = GeometricProposalGenerator()

    # Generate Proposals (with transformation to World Frame)
    proposals = generator.process_lidar_file(
        lidar_path, transform_matrix=transform_matrix
    )

    if not proposals:
        return []

    results = []

    # Parse Ground Truth if available
    gt_objects = []
    if not is_test:
        gt_objects = parse_label_string(row.get("label", ""))

    # Match Proposals and Extract Features
    for prop in proposals:
        # Extract Features
        feat_dict = extract_features_single(prop, sample_token)

        # Initialize Targets
        target_class_id = 0  # Default: Background
        residuals = {k: 0.0 for k in Config.REGRESSION_TARGETS}

        if not is_test:
            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching GT
            for idx, gt_obj in enumerate(gt_objects):
                iou = calc_iou_3d(prop["box"], gt_obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Assign Labels based on IoU Thresholds
            if best_iou >= Config.IOU_POS_THRESH:
                gt_obj = gt_objects[best_gt_idx]
                class_name = gt_obj["class_name"]

                if class_name in Config.CLASS_TO_ID:
                    target_class_id = Config.CLASS_TO_ID[class_name]
                    residuals = compute_residuals(prop["box"], gt_obj["box"])
                else:
                    # Class not in our list, treat as background to avoid confusion
                    target_class_id = 0

            elif best_iou < Config.IOU_NEG_THRESH:
                target_class_id = 0  # Background
            else:
                # Ambiguous region (NEG < IoU < POS)
                # Mark as -1 to filter out during training
                target_class_id = -1

        # Combine all data
        row_data = feat_dict.copy()
        row_data["target_class"] = target_class_id
        row_data.update(residuals)

        # Store proposal box parameters for reconstruction during inference
        row_data["prop_x"] = prop["box"][0]
        row_data["prop_y"] = prop["box"][1]
        row_data["prop_z"] = prop["box"][2]
        row_data["prop_w"] = prop["box"][3]
        row_data["prop_l"] = prop["box"][4]
        row_data["prop_h"] = prop["box"][5]
        row_data["prop_yaw"] = prop["box"][6]

        results.append(row_data)

    return results


class FeatureEngineer:
    """
    Manages the creation of tabular datasets from raw LIDAR and metadata.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def create_dataset(
        self, metadata_path, save_name, load_cached_data=True, is_test=False
    ):
        """
        Main function to process a metadata file into a tabular dataset.

        Args:
            metadata_path (str): Path to the metadata CSV.
            save_name (str): Filename for the cached parquet file.
            load_cached_data (bool): Whether to attempt loading from cache.
            is_test (bool): If True, skips GT matching.

        Returns:
            pd.DataFrame: The processed dataset.
        """
        save_path = os.path.join(self.working_dir, save_name)

        # 1. Check Cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading cached dataset from {save_path}")
            return pd.read_parquet(save_path)

        print(f"Processing dataset from {metadata_path}...")

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # 2a. Load Calibration Data & Precompute Transforms
        # Determine data directory based on metadata filename
        if "test" in os.path.basename(metadata_path):
            data_dir = os.path.join(Config.INPUT_DIR, "test_data")
        else:
            data_dir = os.path.join(Config.INPUT_DIR, "train_data")

        print(f"Loading calibration data from {data_dir}...")
        with open(os.path.join(data_dir, "sample_data.json")) as f:
            sample_data = json.load(f)
        with open(os.path.join(data_dir, "calibrated_sensor.json")) as f:
            calibrated_sensor = json.load(f)
        with open(os.path.join(data_dir, "ego_pose.json")) as f:
            ego_pose = json.load(f)

        # Build Lookups
        # Map filename (basename) -> sample_data_entry
        sd_map = {
            os.path.basename(sd["filename"]): sd
            for sd in sample_data
            if sd["filename"].endswith(".bin")
        }
        calib_map = {item["token"]: item for item in calibrated_sensor}
        ego_map = {item["token"]: item for item in ego_pose}

        # Convert to list of dicts and enrich with transformation matrices
        rows = df_meta.to_dict("records")
        valid_rows = []

        for row in rows:
            lidar_file = os.path.basename(row["lidar_path"])
            if lidar_file in sd_map:
                sd = sd_map[lidar_file]
                calib = calib_map[sd["calibrated_sensor_token"]]
                ego = ego_map[sd["ego_pose_token"]]

                # M_sensor_to_ego
                M_se = get_transform_matrix(calib["translation"], calib["rotation"])
                # M_ego_to_world
                M_ew = get_transform_matrix(ego["translation"], ego["rotation"])
                # M_total = M_ew * M_se
                M_total = M_ew @ M_se

                row["transform_matrix"] = M_total
                valid_rows.append(row)
            else:
                # Fallback if mapping fails (should not happen with valid metadata)
                valid_rows.append(row)

        rows = valid_rows

        # 3. Parallel Processing
        # Use joblib to parallelize the loop over samples
        results_nested = Parallel(n_jobs=Config.NUM_WORKERS, verbose=0)(
            delayed(process_sample_wrapper)(row, is_test)
            for row in tqdm(rows, desc="Extracting Features")
        )

        # Flatten the list of lists
        flat_results = [item for sublist in results_nested for item in sublist]

        if not flat_results:
            print("Warning: No proposals generated from the dataset.")
            # Return empty DF with expected columns to prevent crashes
            return pd.DataFrame()

        # 4. Create DataFrame
        df_features = pd.DataFrame(flat_results)

        # 5. Save to Cache
        print(f"Saving dataset to {save_path}")
        df_features.to_parquet(save_path, index=False)

        return df_features
