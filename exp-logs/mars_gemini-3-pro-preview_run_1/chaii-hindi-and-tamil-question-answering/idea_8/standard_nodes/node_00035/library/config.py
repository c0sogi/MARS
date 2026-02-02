import os
import torch


class Config:
    """
    Configuration for the QA Hindi/Tamil Task.
    Implements Idea 8: Adversarial Multi-Task XLM-R Large with Full-Data Seed Ensemble.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    project_name = "qa_hindi_tamil"
    idea_name = "idea_8"
    seed = 42

    # Debugging
    debug = False  # Set to True to run on a small subset of data
    debug_sample_size = 50  # Number of samples to use in debug mode

    # Ensemble Strategy
    # We train 5 independent models on the full dataset (Train + Val)
    ensemble_seeds = [42, 101, 999, 2023, 12345]

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Input Data (Metadata contains the correct splits)
    input_root = "./metadata"
    raw_input_root = "./input"

    train_path = os.path.join(input_root, "train.csv")
    val_path = os.path.join(input_root, "val.csv")
    test_path = os.path.join(input_root, "test.csv")
    sample_submission_path = os.path.join(raw_input_root, "sample_submission.csv")

    # Working Directory (for artifacts, cache, and outputs)
    working_dir = os.path.join("./working", idea_name)
    output_dir = os.path.join(working_dir, "output")
    cache_dir = os.path.join(working_dir, "cache")
    submission_path = os.path.join(working_dir, "submission", "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "xlm-roberta-large"
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # Sliding Window Configuration
    max_length = 384
    doc_stride = 128
    pad_to_max_length = True

    # Negative Sampling Strategy
    # Retain 100% of positive windows, downsample negatives to this ratio
    negative_positive_ratio = 2.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 8  # Extended training duration
    train_batch_size = 4  # Small batch regularization
    eval_batch_size = 8
    gradient_accumulation_steps = 1

    # Differential Learning Rates (DLR)
    lr_backbone = 1e-5  # Lower rate for the pre-trained backbone
    lr_head = 5e-5  # Higher rate for the task-specific heads

    # Optimizer
    weight_decay = 0.01
    adam_epsilon = 1e-6
    max_grad_norm = 1.0
    warmup_ratio = 0.1

    # Adversarial Training (FGM)
    use_fgm = False
    fgm_epsilon = 1.0
    fgm_param_name = "word_embeddings"  # Parameter to perturb

    # =========================================================================
    # System
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    pin_memory = True

    # =========================================================================
    # Inference / Post-processing
    # =========================================================================
    n_best_size = 20
    max_answer_length = 30

    def __init__(self, **kwargs):
        """
        Initialize Config with optional overrides.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
