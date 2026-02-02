import os
import torch
import random
import numpy as np

# Ensure the working directory exists as per requirements
os.makedirs("./working/idea_4", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Paths
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_4"
    OUTPUT_DIR = "./working/idea_4"  # Where checkpoints are saved
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size of 8 fits comfortably on A100 for Siamese DeBERTa (2 passes per sample)
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16

    # Training duration
    EPOCHS = 8

    # Optimization
    LR_BACKBONE = 1e-5  # Lower learning rate for pre-trained weights
    LR_HEAD = 1e-4  # Higher learning rate for the new interaction head
    WEIGHT_DECAY = 0.01
    EPS = 1e-6
    BETAS = (0.9, 0.999)
    MAX_GRAD_NORM = 1.0

    # Scheduler
    WARMUP_RATIO = 0.1

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # ==========================================
    # Target Labels
    # ==========================================
    # Exact order as per sample_submission.csv
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

    @staticmethod
    def get_optimizer_params(model, lr_backbone=None, lr_head=None, weight_decay=None):
        """
        Constructs parameter groups for the optimizer to apply differential learning rates.

        Args:
            model: The PyTorch model.
            lr_backbone: Learning rate for the DeBERTa backbone.
            lr_head: Learning rate for the custom head/regressor.
            weight_decay: Weight decay coefficient.

        Returns:
            List of parameter group dictionaries.
        """
        lr_backbone = lr_backbone if lr_backbone is not None else Config.LR_BACKBONE
        lr_head = lr_head if lr_head is not None else Config.LR_HEAD
        weight_decay = weight_decay if weight_decay is not None else Config.WEIGHT_DECAY

        # Identify backbone parameters (usually named 'backbone' or 'encoder')
        # In our implementation, we will likely name the transformer part 'backbone'
        optimizer_parameters = []

        # Separate parameters into backbone and head, and apply weight decay logic
        # (no weight decay for bias and LayerNorm)
        no_decay = ["bias", "LayerNorm.weight"]

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine learning rate group
            if "backbone" in name or "deberta" in name:
                lr = lr_backbone
            else:
                lr = lr_head

            # Determine weight decay
            if any(nd in name for nd in no_decay):
                wd = 0.0
            else:
                wd = weight_decay

            optimizer_parameters.append(
                {"params": [param], "lr": lr, "weight_decay": wd}
            )

        return optimizer_parameters
