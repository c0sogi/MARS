import os
import torch


class Config:
    """
    Configuration class for the Metadata-Enhanced DeBERTa-v3-Small Dual-Encoder experiment.
    Acts as the central source of truth for hyperparameters and paths.
    """

    # ==========================================
    # General Settings
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset of data
    debug_sample_size = 100  # Number of samples to use when debug=True

    # ==========================================
    # Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_8/"
    output_dir = "./working/idea_8/"

    # Specific file paths
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")
    submission_path = os.path.join(working_dir, "submission.csv")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    model_name = "microsoft/deberta-v3-small"

    # Tokenizer settings
    max_len = 512  # Maximum sequence length for tokenization

    # Training settings
    epochs = 5
    train_batch_size = 8
    valid_batch_size = 32

    # Optimizer & Scheduler settings
    # Differential learning rates: lower for backbone, higher for new head/embeddings
    lr_backbone = 2e-5
    lr_head = 1e-3
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    scheduler_type = "linear"
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # ==========================================
    # System Settings
    # ==========================================
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Definitions
    # ==========================================
    # 30 Target Labels
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

    # Feature Columns
    text_cols = ["question_title", "question_body", "answer"]
    cat_cols = ["category", "host"]

    def __init__(self):
        # Ensure working directory exists upon instantiation
        os.makedirs(self.working_dir, exist_ok=True)


# Instantiate config to ensure directories are created if imported
config = Config()
