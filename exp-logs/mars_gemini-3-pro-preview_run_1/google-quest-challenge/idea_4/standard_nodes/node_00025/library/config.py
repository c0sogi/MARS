import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the DeBERTa-v3 Dual-Encoder model pipeline.
    Centralizes all hyperparameters, paths, and settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Directories & Paths
    # ==========================================
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Output directories (Write Allowed)
    working_dir = "./working/idea_4"
    submission_dir = "./submission"

    # File Paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output Paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Cache Paths (for deterministic data processing)
    train_cache_path = os.path.join(working_dir, "train_processed.parquet")
    val_cache_path = os.path.join(working_dir, "val_processed.parquet")
    test_cache_path = os.path.join(working_dir, "test_processed.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 512  # Max sequence length for tokenization
    hidden_size = 768  # Hidden size for DeBERTa-v3-base

    # Pooling & Architecture
    num_pooling_layers = (
        4  # Number of last hidden layers to use for Weighted Layer Pooling
    )
    dropout = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 5

    # Batch Sizes
    # Physical batch size is small to fit in GPU memory
    train_batch_size = 4
    valid_batch_size = 16

    # Gradient Accumulation
    # Effective batch size = train_batch_size * gradient_accumulation_steps
    # 4 * 4 = 16 effective batch size
    gradient_accumulation_steps = 4

    # Optimization
    max_grad_norm = 1.0
    weight_decay = 0.01

    # Differential Learning Rates
    lr_backbone = 1e-5  # Lower rate for pre-trained layers
    lr_head = 1e-3  # Higher rate for the new prediction head

    # Scheduler
    warmup_ratio = 0.1

    # ==========================================
    # Target Labels
    # ==========================================
    # The 30 target columns to predict
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

    num_classes = len(target_cols)

    @classmethod
    def setup(cls):
        """
        Sets up the environment:
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Set seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.seed)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
