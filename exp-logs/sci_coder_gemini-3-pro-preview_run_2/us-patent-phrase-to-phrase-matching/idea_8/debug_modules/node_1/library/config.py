import os
import torch


class Config:
    """
    Configuration class for the Phrase Similarity Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Setup
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Metadata generated in previous steps)
    INPUT_DIR = "./metadata"
    TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
    VAL_FILE = os.path.join(INPUT_DIR, "val.csv")
    TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_FILE = "./input/sample_submission.csv"

    # Output Directories
    # Main working directory for this idea iteration
    WORKING_DIR = "./working/idea_8"

    # Sub-directories for organization
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission Directory (Required by task)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 140  # Covers Context + Anchor + Target + Special Tokens safely

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    FOLDS = 5
    EPOCHS = 5

    # Optimization
    # Effective Batch Size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS = 8 * 4 = 32
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    MAX_GRAD_NORM = 1.0

    # Optimizer & Scheduler
    LR = 2e-5
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    SCHEDULER_TYPE = "cosine"
    NUM_WARMUP_STEPS = 0

    # Loss Function
    LABEL_SMOOTHING = 0.1

    # =========================================================================
    # Feature Engineering & Context
    # =========================================================================
    # CPC Section mapping for context enrichment
    # Used to expand single-letter or code contexts into descriptive text
    CPC_SECTIONS = {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
    }

    # Meta-Learner (Stage 2)
    META_MODEL_PATH = os.path.join(MODELS_DIR, "ridge_meta_learner.pkl")
