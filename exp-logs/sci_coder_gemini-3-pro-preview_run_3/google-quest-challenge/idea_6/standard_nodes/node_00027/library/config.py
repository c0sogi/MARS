import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General System Config
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    # Using metadata files as the source of truth for train/test splits
    TRAIN_PATH = "./metadata/train_metadata.csv"
    TEST_PATH = "./metadata/test_metadata.csv"
    SAMPLE_SUB_PATH = "./input/sample_submission.csv"

    # Working directory for caching features, models, and intermediate results
    WORKING_DIR = "./working/idea_6/"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission/"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Architecture & Training Config
    # --------------------------------------------------------------------------
    # Cross-Validation Settings
    N_FOLDS = 4
    GROUP_COL = "question_body"  # For GroupShuffleSplit

    # Input Processing
    MAX_LEN = 512

    # Training Hyperparameters
    # Note: Batch size is kept small for Deberta-Large; gradient accumulation can be used in training loop
    BATCH_SIZE = 4
    EPOCHS = 4
    PATIENCE = 2  # For early stopping
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    GRAD_ACCUM_STEPS = 1

    # Backbone Configurations
    # We define specific settings for each stream in the ensemble
    BACKBONES = {
        "deberta": {
            "name": "microsoft/deberta-v3-large",
            "lr": 1e-5,
            "batch_size": 2,
            "grad_accum_steps": 2,
        },
        "mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "lr": 2e-5,
            "batch_size": 8,
            "grad_accum_steps": 1,
        },
        "roberta": {
            "name": "roberta-base",
            "lr": 2e-5,
            "batch_size": 8,
            "grad_accum_steps": 1,
        },
    }

    # --------------------------------------------------------------------------
    # Target Labels
    # --------------------------------------------------------------------------
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
