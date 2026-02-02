import os
import torch


class Config:
    """
    Configuration for the Custom Ultra-Wide ECA-RepNeXt experiment.
    """

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    # Using idea_36 as the specific working directory for this experiment
    WORKING_DIR = "./working/idea_36"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Hyperparameters ---
    IMAGE_SIZE = (32, 32)
    BATCH_SIZE = 256
    NUM_WORKERS = 4  # Adjusted for standard vCPU availability

    # --- Model Architecture Hyperparameters ---
    MODEL_NAME = "RightSized_ECA_RepNeXt"
    # Right-sized Channel Configuration (Cite solution_lesson_node_00019, solution_lesson_node_00016)
    STAGES_CHANNELS = [32, 64, 128]
    CARDINALITY = 32
    USE_ECA = True
    USE_MULTI_SCALE_HEAD = True

    # --- Training Hyperparameters ---
    # Homogeneous Seed Averaging strategy
    SEEDS = [0, 1, 2, 3, 4]
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    EARLY_STOPPING_PATIENCE = 5

    # --- Device Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_model_save_path(cls, seed):
        """
        Returns the path to save the model checkpoint for a specific seed.
        """
        return os.path.join(cls.WORKING_DIR, f"model_seed_{seed}.pth")


# Automatically setup directories upon import
Config.setup()
