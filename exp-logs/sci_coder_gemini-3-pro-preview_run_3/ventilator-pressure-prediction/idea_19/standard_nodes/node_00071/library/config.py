import os
import torch


class Config:
    """
    Configuration for Idea 19: Wide-Fusion Stabilized Dense-Hybrid Network (WSDH-Net).
    Acts as the single source of truth for paths, hyperparameters, and feature flags.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    EXP_ID = "idea_19"
    INPUT_DIR = "./metadata"
    WORKING_DIR = f"./working/{EXP_ID}"
    SUBMISSION_DIR = "./submission"

    # Input Data (Metadata)
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    VAL_CSV = os.path.join(INPUT_DIR, "validation.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Cache Files (Parquet for speed/size)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")

    # Output Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # Core Physics
    USE_DT = True  # Explicit time delta (dt)
    USE_AREA = True  # Integral of u_in (Volume approximation)
    USE_DELTA_U_IN = True  # Derivative of u_in (Acceleration)

    # Context
    LOOKAHEAD_STEPS = 4  # u_in(t+1) ... u_in(t+4)
    USE_INTERACTION = True  # R*u_in, Area/C

    # Constraints
    EXCLUDE_RAW_TIME = (
        True  # Remove raw monotonic time_step to prevent translation bias
    )

    # ==========================================
    # Model Architecture (WSDH-Net)
    # ==========================================
    # Branch 1: Stabilized Dense Large-Kernel TCN (Resistive Stream)
    TCN_KERNEL_SIZE = 9
    TCN_CHANNELS = [64, 128, 256, 512]  # Increasing capacity
    TCN_DROPOUT = 0.1
    TCN_DILATION = 1  # Strict Dense Convolutions (No dilation)

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.1

    # Fusion Head
    FUSION_HIDDEN_DIM = 1024  # Wide latent integration layer

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128  # Small batch size for generalization
    EPOCHS = 80  # Extended training for hybrid convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD_NORM = 1.0  # Mandatory gradient clipping

    # Optimization
    PATIENCE = 15  # Early stopping patience
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Loss
    MASK_EXPIRATORY = True  # Only calculate loss on inspiratory phase (u_out=0)

    # ==========================================
    # Hardware & Runtime
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def initialize(cls):
        """
        Ensures strict directory safety for the experiment.
        Must be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Print configuration summary
        print(f"Initialized Experiment: {cls.EXP_ID}")
        print(f"Device: {cls.DEVICE}")
        print(f"Working Directory: {cls.WORKING_DIR}")
        print(f"Batch Size: {cls.BATCH_SIZE}, Epochs: {cls.EPOCHS}")
        print(f"Gradient Clipping: {cls.CLIP_GRAD_NORM}")
