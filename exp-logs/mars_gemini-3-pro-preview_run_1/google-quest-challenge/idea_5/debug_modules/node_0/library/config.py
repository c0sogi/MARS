import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run with a small subset of data
    debug_sample_size = 100  # Number of samples to use in debug mode
    num_workers = 4

    # =========================================================================
    # Paths
    # =========================================================================
    # Input paths (Metadata)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output paths
    working_dir = "./working/idea_5/"
    output_model_path = os.path.join(working_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    # Ensure working directory exists (safe to do here or in main script)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Model Settings
    # =========================================================================
    model_name = "microsoft/deberta-v3-small"
    max_len = 512  # Maximum sequence length (dynamic padding will be used up to this)

    # Architecture specifics
    n_dropout_samples = 5  # For Multi-Sample Dropout
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # =========================================================================
    # Training Settings
    # =========================================================================
    epochs = 5
    train_batch_size = 16
    valid_batch_size = 32  # Can be larger as no gradients are stored

    # Differential Learning Rates
    lr_backbone = 2e-5
    lr_head = 1e-3

    # Optimizer (AdamW)
    weight_decay = 0.01
    eps = 1e-6
    betas = (0.9, 0.999)

    # Scheduler
    scheduler_type = "linear"
    warmup_ratio = 0.1

    # Gradient Accumulation (1 means no accumulation, update every batch)
    gradient_accumulation_steps = 1

    # Max grad norm for clipping
    max_grad_norm = 1.0

    # =========================================================================
    # Target Labels
    # =========================================================================
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

    num_targets = len(target_cols)
