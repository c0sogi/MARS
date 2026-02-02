import os
import torch


class Config:
    """
    Configuration class for the StackExchange Question-Answer Labeling task.
    Centralizes file paths, hyperparameters, and system settings.
    """

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache directory for the specific idea implementation
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Vocabulary size for tokenization (top N frequent words)
    VOCAB_SIZE = 20000

    # Maximum sequence length for padding/truncation
    MAX_LEN = 300

    # Dimension of the embedding layer
    EMBED_DIM = 100

    # Dimension of the hidden layers in the MLP head
    HIDDEN_DIM = 128

    # Dropout rate for regularization
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # ==========================================
    # System Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # ==========================================
    # Target Definitions
    # ==========================================
    # List of 30 target columns to predict
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

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
