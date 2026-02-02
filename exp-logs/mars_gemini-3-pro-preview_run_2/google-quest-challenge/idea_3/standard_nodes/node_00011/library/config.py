import os


class Config:
    """
    Configuration class for the Causal-Aware Siamese DeBERTa Network.
    Centralizes hyperparameters, file paths, and target definitions.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    WORKING_DIR = "./working/idea_3/"

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLES = 100

    # ==========================================
    # Data Paths
    # ==========================================
    # Using metadata splits to prevent leakage
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone model
    MODEL_NAME = "microsoft/deberta-v3-base"

    # Input dimensions
    MAX_LEN = 512

    # Training settings
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    EPOCHS = 10
    PATIENCE = 2  # Early stopping patience

    # Optimization (Differential Learning Rates)
    LR_BACKBONE = 1e-5  # Lower rate for pre-trained layers
    LR_HEAD = 1e-4  # Higher rate for custom classification heads
    WEIGHT_DECAY = 0.01

    # ==========================================
    # Target Definitions
    # ==========================================
    # 21 Question-related targets (predicted from Question Head)
    QUESTION_TARGETS = [
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
    ]

    # 9 Answer-related targets (predicted from Answer Interaction Head)
    ANSWER_TARGETS = [
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

    # Combined list of all 30 targets
    TARGET_COLS = QUESTION_TARGETS + ANSWER_TARGETS

    # Target counts
    NUM_QUESTION_TARGETS = len(QUESTION_TARGETS)
    NUM_ANSWER_TARGETS = len(ANSWER_TARGETS)
    NUM_TOTAL_TARGETS = len(TARGET_COLS)

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
