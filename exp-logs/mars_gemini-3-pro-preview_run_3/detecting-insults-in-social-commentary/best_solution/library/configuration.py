import os
import torch


class Config:
    # ==========================================
    # General Setup
    # ==========================================
    PROJECT_NAME = "insult_detection_ensemble"
    IDEA_NAME = "idea_8"
    SEEDS = [42, 43, 44]  # Seeds for ensemble averaging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # ==========================================
    # Data Paths
    # ==========================================
    # Using the metadata generated in the metadata step
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Output Directories
    # ==========================================
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Hyperparameters
    # ==========================================
    MAX_LEN = 160

    # Effective Batch Size = TRAIN_BATCH_SIZE * GRAD_ACC_STEPS = 8 * 4 = 32
    TRAIN_BATCH_SIZE = 8
    GRAD_ACC_STEPS = 4
    VALID_BATCH_SIZE = 16

    # Optimization
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Regularization
    DROPOUT = 0.2

    # ==========================================
    # Model Configurations
    # ==========================================
    # Heterogeneous Ensemble: RoBERTa-Large + DeBERTa-v3-Large
    # Cite solution_lesson_node_00034: Heterogeneous Ensembling with Diverse Architectures
    # Cite solution_lesson_node_00010: Efficiency and Stability via Partial Layer Freezing
    MODEL_CONFIGS = [
        {
            "model_name": "roberta-large",
            "tokenizer_path": "roberta-large",
            "epochs": 4,  # Increased epochs for better convergence
            "freeze_layers": 6,
            "dropout": 0.2,
        },
        {
            "model_name": "microsoft/deberta-v3-large",
            "tokenizer_path": "microsoft/deberta-v3-large",
            "epochs": 4,  # Increased epochs (previously 2 was under-trained)
            "freeze_layers": 6,
            "dropout": 0.2,
        },
    ]

    # Pseudo-Labeling Thresholds
    PSEUDO_LABEL_CONF_HIGH = 0.95
    PSEUDO_LABEL_CONF_LOW = 0.05
