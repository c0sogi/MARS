import os


class Config:
    """
    Configuration for the Dual-Branch Deep Averaging Network (DAN) pipeline.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a smaller subset for debugging
    DEBUG_SIZE = 1000

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Data Files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Text Processing
    # ==========================================
    VOCAB_SIZE = 20000
    # Max sequence length for Question (Title + Body)
    MAX_LEN_Q = 256
    # Max sequence length for Answer
    MAX_LEN_A = 256
    TOKENIZER_LOWER = True

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    PATIENCE = 3  # Early stopping patience

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

    @staticmethod
    def setup():
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
