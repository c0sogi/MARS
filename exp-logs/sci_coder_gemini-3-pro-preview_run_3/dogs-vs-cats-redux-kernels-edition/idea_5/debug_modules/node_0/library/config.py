import os
import torch


class Config:
    # --- General Experiment Settings ---
    PROJECT_NAME = "DogVsCat_Stacked_Ensemble"
    IDEA_ID = "idea_5"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Ensure working directory is specific to this idea
    WORKING_DIR = os.path.join("./working", IDEA_ID)

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Data Paths ---
    # We will likely combine train and val CSVs for full 5-fold CV
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # --- Model Hyperparameters ---
    # Image Size: 320x320 as per Idea 5 strategy
    IMG_SIZE = 320

    # Batch Size: 64 to fit 320x320 on A100 GPU
    BATCH_SIZE = 64

    # Training Duration: 6-8 epochs suggested
    EPOCHS = 7

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Cross Validation
    N_FOLDS = 5

    # --- Architecture Definitions ---
    # Using timm compatible model names
    # 1. ResNet-101 (Robust CNN backbone)
    # 2. ConvNeXt-Small (Modern Transformer-inspired backbone)
    MODEL_CONFIGS = [
        {"name": "resnet101.a1_in1k", "type": "cnn", "pretrained": True},
        {
            "name": "convnext_small.fb_in1k",
            "type": "transformer_hybrid",
            "pretrained": True,
        },
    ]

    # Stacking Meta-Learner
    META_LEARNER_PARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"--- Configuration: {cls.IDEA_ID} ---")
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Models: {[m['name'] for m in cls.MODEL_CONFIGS]}")
        print(f"Folds: {cls.N_FOLDS}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("-" * 30)
