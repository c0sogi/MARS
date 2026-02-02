import os


class Config:
    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # For data loading

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Paths (Metadata)
    # --------------------------------------------------------------------------
    TRAIN_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --------------------------------------------------------------------------
    # Column Definitions
    # --------------------------------------------------------------------------
    ID_COL = "qa_id"
    TEXT_COLS = ["question_title", "question_body", "answer"]

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

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Backbone 1: MPNet (Semantic Anchor)
    MODEL_1_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Shared Tokenizer/Model Config
    MAX_LENGTH = 512
    HIDDEN_DROPOUT_PROB = 0.1

    # Training Config
    TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = (
        1  # Effective batch size = 8 (Cite solution_lesson_node_00007)
    )
    EVAL_BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 6
    PATIENCE = 2  # Early stopping patience based on Validation Loss

    # Ridge Regression Config
    RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

    # --------------------------------------------------------------------------
    # Caching Paths (Intermediate Features)
    # --------------------------------------------------------------------------
    # Model 1 Features (Updated paths to force re-training/extraction)
    M1_TRAIN_FEATS_PATH = os.path.join(WORKING_DIR, "m1_train_features_v2.npy")
    M1_VAL_FEATS_PATH = os.path.join(WORKING_DIR, "m1_val_features_v2.npy")
    M1_TEST_FEATS_PATH = os.path.join(WORKING_DIR, "m1_test_features_v2.npy")
    M1_MODEL_PATH = os.path.join(WORKING_DIR, "m1_finetuned_v2.pth")

    # Targets (Cached for convenience)
    TRAIN_TARGETS_PATH = os.path.join(WORKING_DIR, "train_targets.npy")
    VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.npy")

    # Final Ridge Model
    RIDGE_MODEL_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")

    # Final Submission File
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
