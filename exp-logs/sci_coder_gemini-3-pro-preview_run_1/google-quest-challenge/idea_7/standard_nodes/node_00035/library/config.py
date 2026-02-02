import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths
    # ==========================================
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUB_PATH = "./input/sample_submission.csv"

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_7/"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LEN = 512
    HIDDEN_SIZE = 768  # Hidden size for DeBERTa-v3-small

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 5

    # Differential Learning Rates
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-3

    WEIGHT_DECAY = 0.01
    AUX_LOSS_WEIGHT = 0.5  # Alpha for auxiliary loss

    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Target Columns
    # ==========================================
    # All 30 targets
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

    # 21 Question-specific targets for Auxiliary Head
    QUESTION_TARGET_COLS = [
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


def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
