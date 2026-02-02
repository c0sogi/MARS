import os
import torch

# -----------------------------------------------------------------------------
# Target Columns
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
class PathConfig:
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mpnet_cross_encoder.pth")
    RIDGE_SAVE_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (Prefixes or Directories)
    # We will save processed features here.
    # e.g. ./working/idea_3/train_features.npy
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.npy")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.npy")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.npy")

    TRAIN_TARGETS_CACHE = os.path.join(WORKING_DIR, "train_targets.npy")
    VAL_TARGETS_CACHE = os.path.join(WORKING_DIR, "val_targets.npy")

    @staticmethod
    def setup_directories():
        """Ensure necessary directories exist."""
        os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)
        os.makedirs(PathConfig.SUBMISSION_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
class ModelConfig:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    max_len = 512
    hidden_size = 768  # Hidden size for mpnet-base
    dropout = 0.1
    num_labels = len(TARGET_COLS)


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
class TrainConfig:
    seed = 42
    epochs = 8
    batch_size = 8
    grad_acc_steps = 2
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    num_workers = 4

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ridge Regression Params (Stage 2)
    ridge_alphas = [0.1, 1.0, 10.0, 100.0]


# Ensure directories exist upon import
PathConfig.setup_directories()
