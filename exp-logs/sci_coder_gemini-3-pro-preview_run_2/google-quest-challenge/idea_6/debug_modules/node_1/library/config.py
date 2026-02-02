import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of rows to use if DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    HIDDEN_SIZE = 768

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 6

    # Differential Learning Rates
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-4

    WEIGHT_DECAY = 0.01
    PATIENCE = 3  # Early stopping patience
    NUM_WORKERS = 4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Features & Targets
    # ==========================================
    # Categorical features for Metadata Fusion
    CAT_COLS = ["category", "host"]

    # List of 30 Target Columns
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
