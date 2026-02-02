import os
import torch


class Config:
    """
    Configuration class for Siamese DeBERTa with Granular Co-Attention Fusion.
    Centralizes all hyperparameters, paths, and settings.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging
    debug_sample_size = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # Paths
    # ==========================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory specific to Idea 8
    working_dir = "./working/idea_8"
    output_dir = "./working/idea_8"  # Where model checkpoints are saved

    # Submission paths
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Data file paths (using metadata splits)
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")

    # ==========================
    # Model Hyperparameters
    # ==========================
    model_name = "microsoft/deberta-v3-base"

    # Input Sequence Lengths (Siamese Streams)
    max_len_q = 512  # For [CLS] Title [SEP] Body [SEP]
    max_len_a = 512  # For [CLS] Answer [SEP]

    # Architecture
    hidden_size = 768
    dropout = 0.1
    num_targets = 30

    # ==========================
    # Training Hyperparameters
    # ==========================
    epochs = 6
    train_batch_size = 4  # Adjusted for A100 memory with dual 512 context + CoAttention
    valid_batch_size = 16
    accumulation_steps = 2

    # Optimizer (AdamW)
    lr_backbone = 1e-5
    lr_head = 1e-4
    weight_decay = 0.01
    llrd_decay = 0.9
    max_grad_norm = 1.0

    # Scheduler (Cosine Annealing with Warmup)
    warmup_ratio = 0.1

    # ==========================
    # Features & Targets
    # ==========================
    text_cols = ["question_title", "question_body", "answer"]
    cat_cols = ["category", "host"]

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

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
