import os
import torch


class Config:
    """
    Global configuration for the Convex-Densified Selective-Topology LDA pipeline.
    Defines hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False, limit_dataset=None):
        """
        Initialize the configuration.

        Args:
            debug (bool): If True, enables debug mode (verbose logging).
            limit_dataset (int, optional): If provided, limits the number of samples
                                           processed for debugging purposes.
        """
        # ==========================================
        # System & Environment
        # ==========================================
        self.SEED = 42
        self.N_JOBS = 12
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEBUG = debug
        self.LIMIT_DATASET = limit_dataset

        # ==========================================
        # Directory Paths
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_36"
        self.SUBMISSION_DIR = "./submission"
        self.IMAGES_DIR = os.path.join(self.INPUT_DIR, "images")

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # ==========================================
        # File Paths
        # ==========================================
        self.TRAIN_METADATA_PATH = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_METADATA_PATH = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_METADATA_PATH = os.path.join(self.METADATA_DIR, "test.csv")
        self.SAMPLE_SUBMISSION_PATH = os.path.join(
            self.INPUT_DIR, "sample_submission.csv"
        )
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # ==========================================
        # Model Architecture (Dual-Stream)
        # ==========================================
        # Global Geometry Stream: DINOv2 (ViT-Large)
        self.MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"

        # Local Texture Stream: ConvNeXt Large
        self.MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

        self.IMG_SIZE = 224
        self.BATCH_SIZE = 32

        # ==========================================
        # Manifold Densification Strategy
        # ==========================================
        # We extract 12 rotations to cover the feature manifold
        self.N_ROTATIONS = 12
        self.ROTATION_ANGLES = [
            i * (360 // self.N_ROTATIONS) for i in range(self.N_ROTATIONS)
        ]

        # Orthogonal Centroid Logic:
        # We generate 3 Primary Centroids by averaging 4 orthogonal views each.
        # Indices correspond to the 12 rotations [0..11].
        # C1: 0, 90, 180, 270 degrees
        # C2: 30, 120, 210, 300 degrees
        # C3: 60, 150, 240, 330 degrees
        self.PRIMARY_CENTROIDS = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

        # Interpolation weight for Secondary Centroids (Convex Hull)
        self.INTERPOLATION_ALPHA = 0.5

        # ==========================================
        # Feature Engineering & Dimensionality
        # ==========================================
        # Retain 99% variance in PCA for visual streams
        self.PCA_VARIANCE = 0.99

        # Tabular feature groups
        self.TABULAR_PREFIXES = ["margin", "shape", "texture"]

        # ==========================================
        # Training Strategy (LDA Ensemble)
        # ==========================================
        self.N_FOLDS = 10
