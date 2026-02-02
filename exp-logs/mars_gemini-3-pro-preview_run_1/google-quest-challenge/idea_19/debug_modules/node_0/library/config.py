import os
import torch


class Config:
    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "roberta-base"
    MAX_LEN = 512

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16  # Can be larger as no gradients are stored
    ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16

    # Differential Learning Rates
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-3

    # Phantom Scheduling Strategy
    PHANTOM_EPOCHS = 7  # Scheduler thinks we train for 7 epochs
    ACTUAL_EPOCHS = 3  # We stop after 3 epochs

    # General
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching (Idea 19 as per prompt context)
    WORKING_DIR = "./working/idea_19"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

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

    NUM_LABELS = len(TARGET_COLS)

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
