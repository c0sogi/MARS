import os
import torch


class Config:
    """
    Global configuration for the Topology-Aware Stacking of Domain-Adapted Cross-Encoders pipeline.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 4
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (ensure these exist via metadata generation script)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Configurations
    # --------------------------------------------------------------------------
    # Stream A: Heavyweight Backbone (State-of-the-art NLU)
    MODEL_A_NAME = "microsoft/deberta-v3-large"

    # Stream B: Efficient Baseline (Stable, good structure understanding)
    MODEL_B_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Domain Adaptive Pre-Training (DAPT) Base
    # We typically adapt the larger model to the domain vocabulary
    MLM_MODEL_NAME = MODEL_A_NAME

    # Tokenizer Settings
    MAX_LEN = 512

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Batch sizes optimized for A100 40GB
    TRAIN_BATCH_SIZE = 4  # Small physical batch size for DeBERTa-Large
    VALID_BATCH_SIZE = 16  # Larger for inference/validation

    EPOCHS = 3
    LEARNING_RATE = 1e-5  # Low LR for fine-tuning large models
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    GRAD_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16

    # Early Stopping
    PATIENCE = 3  # Stop if validation loss doesn't improve

    # Ridge Regression Head Params
    RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]  # Regularization strengths for RidgeCV

    # --------------------------------------------------------------------------
    # Target Columns (Topology Definition)
    # --------------------------------------------------------------------------
    # Full list of 30 targets
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

    # Subset: Question-related targets (21 cols)
    # Dependent ONLY on question features ($h_Q$)
    QUESTION_TARGETS = [col for col in TARGET_COLS if col.startswith("question_")]

    # Subset: Answer-related targets (9 cols)
    # Dependent on interaction features ($h_{CLS}, h_Q, h_A, h_{diff}$)
    ANSWER_TARGETS = [col for col in TARGET_COLS if col.startswith("answer_")]

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Initialize the workspace.
        Creates necessary directories for caching and submissions.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"[Config] Working Directory: {cls.WORKING_DIR}")
        print(f"[Config] Submission Directory: {cls.SUBMISSION_DIR}")
        print(f"[Config] Device: {cls.DEVICE}")
