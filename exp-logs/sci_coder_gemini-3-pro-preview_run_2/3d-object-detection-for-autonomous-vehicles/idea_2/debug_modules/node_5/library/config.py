import os


class Config:
    """
    Configuration class for the Cluster-and-Classify 3D Object Detection Pipeline.
    """

    # ==========================================
    # 1. File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate steps (e.g., processed clusters)
    WORKING_DIR = "./working/idea_2"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Global Settings & Reproducibility
    # ==========================================
    SEED = 42

    # Debugging / Development
    # Set to a small number (e.g., 500) to speed up development by using a subset of data
    # Set to None to use the full dataset
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # 3. LiDAR Preprocessing (Sensor Frame)
    # ==========================================
    # Region of Interest (ROI) in meters relative to the sensor
    # Points outside this box are discarded before processing
    X_MIN, X_MAX = -100.0, 100.0
    Y_MIN, Y_MAX = -100.0, 100.0
    Z_MIN, Z_MAX = -5.0, 10.0  # Loose Z bounds to capture ground and tall objects

    # ==========================================
    # 4. Ground Removal (RANSAC)
    # ==========================================
    # Distance threshold: points within this distance to the plane are considered ground
    RANSAC_DIST_THRESH = 0.25
    # Number of iterations for plane fitting
    RANSAC_ITERATIONS = 100

    # ==========================================
    # 5. Clustering (DBSCAN)
    # ==========================================
    # The maximum distance between two samples for one to be considered as in the neighborhood of the other
    DBSCAN_EPS = 0.75
    # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    DBSCAN_MIN_SAMPLES = 5

    # Minimum number of points required to keep a cluster for feature extraction
    MIN_CLUSTER_POINTS = 10

    # ==========================================
    # 6. Target Assignment (Training)
    # ==========================================
    # Thresholds to match a generated cluster to a ground truth bounding box
    # If a cluster center is within this distance of a GT center, it is a positive match
    MATCH_DIST_THRESHOLD = 2.0  # meters

    # ==========================================
    # 7. Model Hyperparameters (XGBoost)
    # ==========================================
    # Parameters for the Classifier (Background vs Object Class)
    XGB_CLF_PARAMS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",  # Multiclass classification
        "eval_metric": "mlogloss",
        "tree_method": "hist",  # Faster training
        "n_jobs": 12,  # Use available vCPUs
        "random_state": SEED,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Parameters for the Regressor (Bounding Box Refinement)
    # Predicts: [center_x, center_y, center_z, width, length, height, yaw]
    XGB_REG_PARAMS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": 12,
        "random_state": SEED,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # ==========================================
    # 8. Class Definitions
    # ==========================================
    # Classes derived from EDA
    CLASSES = [
        "car",
        "other_vehicle",
        "pedestrian",
        "bicycle",
        "truck",
        "bus",
        "motorcycle",
        "animal",
        "emergency_vehicle",
    ]

    # Mapping from class name to integer ID
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
    # Mapping from integer ID to class name
    ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

    # Total number of object classes (excluding background)
    NUM_CLASSES = len(CLASSES)

    # ==========================================
    # 9. Inference / Submission
    # ==========================================
    # Minimum confidence score to include a prediction in the submission
    CONF_THRESHOLD = 0.3

    # Default confidence value if model doesn't output probabilities (fallback)
    DEFAULT_CONFIDENCE = 1.0
