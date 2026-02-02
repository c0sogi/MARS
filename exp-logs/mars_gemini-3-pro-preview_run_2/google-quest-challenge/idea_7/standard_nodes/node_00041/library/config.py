import os


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Metadata file paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512  # Max sequence length for both Question and Answer streams
    HIDDEN_SIZE = 768
    DROPOUT = 0.1
    NUM_LAYERS_TO_AGGREGATE = 4  # For Layer-Wise Dynamic Weighting

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 8
    TRAIN_BATCH_SIZE = 4  # Adjusted for A100 40GB with Siamese 512+512 context
    VALID_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4

    # Differential Learning Rates
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-4

    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 3

    # AWP Config
    USE_AWP = True
    AWP_START_EPOCH = 2
    AWP_LR = 1.0
    AWP_EPS = 1e-2

    # ==========================================
    # Data Columns
    # ==========================================
    # Text Features
    TEXT_COLS = ["question_title", "question_body", "answer"]

    # Categorical Metadata Features
    CAT_COLS = ["category", "host"]

    # Target Labels (30 columns)
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
