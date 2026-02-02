import os
import torch


class Config:
    # Random Seed
    SEED = 42

    # Data Paths
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Model Checkpoints
    WORKING_DIR = "./working/idea_34"
    os.makedirs(WORKING_DIR, exist_ok=True)

    PROCESSED_TRAIN_PATH = os.path.join(WORKING_DIR, "train_data.npz")
    PROCESSED_VAL_PATH = os.path.join(WORKING_DIR, "val_data.npz")
    PROCESSED_TEST_PATH = os.path.join(WORKING_DIR, "test_data.npz")
    SCALERS_PATH = os.path.join(WORKING_DIR, "scalers.npz")

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Data Processing Hyperparameters
    K_NEIGHBORS = 12  # Number of nearest neighbors for local context
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    # Feature Dimensions
    # Atomic Stream: One-hot(4) + Coords(3) + d_min(1) + d_mean(1) + LocalStoich(4)
    ATOMIC_FEATURE_DIM = 4 + 3 + 1 + 1 + 4
    # Global Stream: Lattice(6) + Vol(1) + Density(1) + Stoich(3) + TotalAtoms(1)
    GLOBAL_FEATURE_DIM = 6 + 1 + 1 + 3 + 1

    # Model Architecture
    ATOM_HIDDEN_DIM = 512  # Wide MLP for atomic stream
    GLOBAL_HIDDEN_DIM = 256  # High-capacity MLP for global stream
    FUSION_HIDDEN_DIM = 256  # Dimension after fusing streams
    DROPOUT_RATE = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler & Early Stopping
    PATIENCE = 20
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        print(f"Config Loaded:")
        print(f"  Device: {Config.DEVICE}")
        print(f"  K Neighbors: {Config.K_NEIGHBORS}")
        print(f"  Atomic Feature Dim: {Config.ATOMIC_FEATURE_DIM}")
        print(f"  Global Feature Dim: {Config.GLOBAL_FEATURE_DIM}")
        print(
            f"  Model: AtomHidden={Config.ATOM_HIDDEN_DIM}, GlobalHidden={Config.GLOBAL_HIDDEN_DIM}"
        )
        print(
            f"  Training: Batch={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}, Epochs={Config.NUM_EPOCHS}"
        )
