import os
import random
import torch
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # Model Architecture
    model_name = "roberta-base"
    max_len = 512
    hidden_size = 768  # Standard for roberta-base

    # Training Hyperparameters
    train_batch_size = 8
    valid_batch_size = 16
    accum_steps = 2  # Effective batch size = 8 * 2 = 16
    epochs = 3
    phantom_epochs = 7  # For scheduler decay profile

    # Differential Learning Rates
    lr_head = 1e-3
    lr_backbone = 2e-5
    weight_decay = 0.01

    # Environment
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    # Input paths (Read-only)
    INPUT_DIR = "./input"
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUB_PATH = "./input/sample_submission.csv"

    # Output paths (Working directory)
    WORKING_DIR = "./working/idea_23"
    OUTPUT_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Target Labels (30 columns)
    target_cols = [
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

    num_labels = len(target_cols)

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
