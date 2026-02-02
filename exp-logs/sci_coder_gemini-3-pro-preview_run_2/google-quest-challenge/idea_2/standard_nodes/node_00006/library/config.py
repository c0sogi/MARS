import os
import torch


class Config:
    """
    Configuration class for the Siamese DistilRoBERTa pipeline.
    Stores file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Files (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "distilroberta-base"
    MAX_LENGTH = 512
    HIDDEN_SIZE = 768  # Standard for distilroberta-base
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    # Batch Sizes (A100 40GB allows decent batch sizes)
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 32

    # Optimization
    BACKBONE_LR = 2e-5  # Lower learning rate for pre-trained weights
    HEAD_LR = 1e-3  # Higher learning rate for the new interaction head
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

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

    @classmethod
    def create_dirs(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
