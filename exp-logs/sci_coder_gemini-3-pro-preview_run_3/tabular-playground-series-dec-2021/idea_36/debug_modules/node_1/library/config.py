import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_36"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
SEED = 42
TARGET_COL = "Cover_Type"
ID_COL = "Id"
NUM_CLASSES = 7  # Classes are 1-7


# ==========================================
# Configuration Class
# ==========================================
class Config:
    """
    Central configuration class for the Deeply-Supervised Asymmetric Parallel Vector-DCN-ResNet.
    Allows for dynamic overrides (e.g., debug mode).
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.seed = SEED

        # --------------------------------------
        # Paths
        # --------------------------------------
        self.input_dir = INPUT_DIR
        self.metadata_dir = METADATA_DIR
        self.working_dir = WORKING_DIR
        self.submission_dir = SUBMISSION_DIR

        self.train_path = TRAIN_PATH
        self.val_path = VAL_PATH
        self.test_path = TEST_PATH
        self.sample_submission_path = SAMPLE_SUBMISSION_PATH
        self.submission_path = SUBMISSION_PATH

        # --------------------------------------
        # Feature Engineering Flags
        # --------------------------------------
        # Controls specific augmentations defined in the idea
        self.use_aspect_trig = True  # Aspect_Sin, Aspect_Cos
        self.use_dist_hydro_euclidean = True  # Sqrt(H^2 + V^2)
        self.use_abs_hydro_elevation = True  # Elevation - Vertical_Dist
        self.use_mean_dist_amenities = True  # Mean(Hydro, Road, Fire)

        # --------------------------------------
        # Model Architecture
        # --------------------------------------
        # Deeply-Supervised Asymmetric Parallel Vector-DCN-ResNet
        self.dcn_layers = 3  # Asymmetric depth (Lesson 00071)
        self.resnet_blocks = 5  # Scaled depth (Lesson 00054)
        self.aux_loss_weight = 0.3  # Auxiliary supervision weight
        self.dropout_rate = 0.2  # Regularization (Lesson 00056)

        # Internal dimensions
        self.resnet_width = 256
        self.dcn_projection_dim = 128  # Dimension for vector-based cross layers

        # --------------------------------------
        # Training Hyperparameters
        # --------------------------------------
        self.batch_size = 4096  # (Lesson 00022)
        self.epochs = 60  # Fixed budget
        self.learning_rate = 1e-3  # Base LR for AdamW
        self.weight_decay = 1e-2  # Decoupled weight decay

        # Scheduler (ReduceLROnPlateau)
        self.scheduler_factor = 0.1  # Aggressive decay (Lesson 00068)
        self.scheduler_patience = 5
        self.min_lr = 1e-6

        # Early Stopping
        self.early_stopping_patience = 10

        # Hardware
        self.num_workers = 4

        # --------------------------------------
        # Debug Overrides
        # --------------------------------------
        if self.debug:
            self.epochs = 2
            self.batch_size = 1024
            self.early_stopping_patience = 1

    def __repr__(self):
        return (
            f"Config(debug={self.debug}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, dcn_layers={self.dcn_layers}, "
            f"resnet_blocks={self.resnet_blocks})"
        )
