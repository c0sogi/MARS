import os


class Config:
    # ==========================================
    # Paths
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Input files (using metadata splits as requested)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    # Ensuring the working directory exists is handled by the processing scripts,
    # but we define the path here.
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "distilroberta-base"
    MAX_LEN = 512  # Max length for each branch (Question and Answer)
    HIDDEN_SIZE = 768  # Hidden size of distilroberta-base
    DROPOUT_PROB = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 5

    # Differential Learning Rates
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-3

    # Optimizer settings
    WEIGHT_DECAY = 0.01
    EPS = 1e-8

    # Scheduler settings
    WARMUP_RATIO = 0.1

    # Early Stopping
    PATIENCE = 3

    # ==========================================
    # Data & Targets
    # ==========================================
    # List of 30 target columns in the specific order required for submission
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
