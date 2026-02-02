import os


class Config:
    # Reproducibility
    SEED = 42

    # File Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Processing
    IMAGE_SIZE = 224

    # Global Statistics for Min-Max Scaling (Derived from Data Analysis)
    # Band 1 (HH) - Min: -45.5944, Max: 32.1806
    # Band 2 (HV) - Min: -45.6555, Max: 17.8628
    GLOBAL_STATS = {
        "band_1": {"min": -45.5944, "max": 32.1806},
        "band_2": {"min": -45.6555, "max": 17.8628},
    }

    # Model Hyperparameters
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1

    # Training Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.05

    # Scheduler & Early Stopping
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.1
    EARLY_STOPPING_PATIENCE = 8

    # Hardware
    NUM_WORKERS = 4

    @classmethod
    def make_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
