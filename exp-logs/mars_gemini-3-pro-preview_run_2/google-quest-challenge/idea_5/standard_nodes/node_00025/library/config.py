import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4

    # ==========================================
    # Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_5"

    # Data paths using the generated metadata
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output paths
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # ==========================================
    # Model Configuration
    # ==========================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 512
    hidden_dropout_prob = 0.1

    # ==========================================
    # Training Configuration
    # ==========================================
    epochs = 6
    train_batch_size = 8
    valid_batch_size = 16

    # Differential Learning Rates
    lr_backbone = 1e-5
    lr_head = 1e-4

    weight_decay = 0.01
    max_grad_norm = 1.0

    # Scheduler
    scheduler = "CosineAnnealingWarmRestarts"
    T_0 = 6  # Cycle length matching epochs
    min_lr = 1e-6

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Target Columns
    # ==========================================
    # 21 Question-related targets
    question_targets = [
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

    # 9 Answer-related targets
    answer_targets = [
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

    # Combined target list (30 columns)
    target_cols = question_targets + answer_targets
    num_labels = len(target_cols)
