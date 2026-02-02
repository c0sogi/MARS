import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Metadata Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths (Parquet format preferred over pickle)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "distilroberta-base"

    # Dual-Encoder Setup: Independent max lengths
    MAX_LEN_Q = 512  # Question Title + Body
    MAX_LEN_A = 512  # Answer

    HIDDEN_SIZE = 768
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32

    # Differential Learning Rates
    LR_HEAD = 1e-3  # Higher LR for the initialized head/pooling
    LR_BACKBONE = 2e-5  # Lower LR for the pre-trained backbone

    WEIGHT_DECAY = 0.01

    # Training Schedule
    NUM_EPOCHS = 5
    WARMUP_EPOCHS = 1  # Epochs where backbone is frozen

    # ==========================================
    # Target Labels
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
        """Sets the seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def create_dirs(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
