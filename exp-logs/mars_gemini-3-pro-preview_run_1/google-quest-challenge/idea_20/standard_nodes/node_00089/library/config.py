import os
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Data Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "roberta-base"
    HIDDEN_SIZE = 768
    NUM_TARGETS = 30

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

    # Architecture specifics for Shared-Bottom Multi-Branch
    # Layers 0-9 are shared, 10-11 are split
    SHARED_LAYER_COUNT = 10

    # Initialization for Head
    INIT_MEAN = 0.0
    INIT_STD = 0.02

    # ==========================================
    # Data Processing
    # ==========================================
    MAX_LEN = 512
    TOKENIZER_PATH = "roberta-base"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Batch Size Strategy
    # Physical batch size 8 * Accumulation 2 = Effective 16
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2

    # Learning Rates (Differential)
    LR_HEAD = 1e-3
    LR_BACKBONE = 2e-5

    # Phantom Scheduling Strategy
    # Train for 3 epochs, but schedule decay as if training for 7
    EPOCHS = 3
    SCHEDULER_EPOCHS = 7

    # Head Warmup
    # Freeze backbone for the first epoch
    WARMUP_EPOCHS = 1

    # Regularization
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # Hardware & Logging
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRINT_FREQ = 100

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set environment variables for reproducibility where possible
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
