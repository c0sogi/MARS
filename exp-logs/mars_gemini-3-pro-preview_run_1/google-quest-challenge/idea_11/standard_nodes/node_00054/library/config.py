import os
import torch


class Config:
    """
    Global configuration for the Question-Answer Labeling task.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"

    # Create working directory immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data paths (using metadata splits as required)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching paths for processed data
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "distilroberta-base"
    HIDDEN_SIZE = 768  # Hidden dimension for distilroberta-base
    REINIT_LAYERS = 2  # Number of top transformer layers to re-initialize

    # ==========================================
    # Data Processing
    # ==========================================
    MAX_LEN = 512

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 5

    # Differential Learning Rates
    LR_HIGH = 1e-3  # For Residual Fusion Head and Re-initialized layers
    LR_LOW = 2e-5  # For frozen/pretrained backbone layers

    WEIGHT_DECAY = 0.01

    # ==========================================
    # Target Definitions
    # ==========================================
    TARGET_COLS = [
        "question_asker_intent_understanding",
        "question_body_critical",
        "question_conversational",
        "question_expect_short_answer",
        "question_fact_seeking",
        "question_has_commonly_accepted_answer",
        "question_interestingness_others",
        "question_interestingness_self",
        "question_multi_intent",
        "question_not_really_a_question",
        "question_opinion_seeking",
        "question_type_choice",
        "question_type_compare",
        "question_type_consequence",
        "question_type_definition",
        "question_type_entity",
        "question_type_instructions",
        "question_type_procedure",
        "question_type_reason_explanation",
        "question_type_spelling",
        "question_well_written",
        "answer_helpful",
        "answer_level_of_information",
        "answer_plausible",
        "answer_relevance",
        "answer_satisfaction",
        "answer_type_instructions",
        "answer_type_procedure",
        "answer_type_reason_explanation",
        "answer_well_written",
    ]

    NUM_TARGETS = len(TARGET_COLS)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across all libraries.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
