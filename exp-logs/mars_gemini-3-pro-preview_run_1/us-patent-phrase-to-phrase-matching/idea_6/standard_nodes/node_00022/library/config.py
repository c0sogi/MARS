import os
import torch


class Config:
    """
    Configuration class for the Semantic Similarity Model (DeBERTa-v3-Large).
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================
    # General Settings
    # ==========================
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging
    debug_sample_size = 100  # Number of samples to use in debug mode
    num_workers = 4

    # ==========================
    # File Paths
    # ==========================
    # Root directories
    input_root = "./input"
    metadata_root = "./metadata"
    working_dir = "./working/idea_6"

    # Input Data
    # Using metadata splits as the primary source
    train_path = os.path.join(metadata_root, "train.csv")
    val_path = os.path.join(metadata_root, "val.csv")
    test_path = os.path.join(metadata_root, "test.csv")

    # Auxiliary Data
    # CPC descriptions for context injection
    cpc_description_path = os.path.join(input_root, "description.md")
    sample_submission_path = os.path.join(input_root, "sample_submission.csv")

    # Output Paths
    output_dir = working_dir
    model_dir = os.path.join(output_dir, "models")
    predictions_dir = os.path.join(output_dir, "predictions")
    submission_path = "./submission/submission.csv"

    # ==========================
    # Model Architecture
    # ==========================
    model_name = "microsoft/deberta-v3-large"
    num_classes = 1  # Regression task

    # Multi-Sample Dropout (MSD) Settings
    use_msd = True
    msd_samples = 5
    msd_dropout = 0.1

    # ==========================
    # Tokenizer & Data
    # ==========================
    # Max length for [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    # Anchor/Target are short, but Context Hierarchy can be verbose.
    max_length = 256

    # ==========================
    # Training Hyperparameters
    # ==========================
    folds = 5
    epochs = 4

    # Batch Sizes (A100 40GB allows for larger batches)
    train_batch_size = 8
    valid_batch_size = 16
    gradient_accumulation_steps = 1

    # Optimization
    max_grad_norm = 10.0
    weight_decay = 0.01
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # Layer-wise Learning Rate Decay (LLRD)
    # Lower layers (closer to input) get smaller LRs to preserve pre-trained features
    learning_rate = 2e-5  # Base LR for the backbone
    head_lr = 1e-4  # Higher LR for the regression head
    llrd_decay = 0.9  # Decay factor for each layer

    # Precision
    fp16 = True

    # ==========================
    # Hardware
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)

        # Set deterministic behavior for reproducibility
        os.environ["PYTHONHASHSEED"] = str(cls.seed)
