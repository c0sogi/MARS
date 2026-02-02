import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-v3-Base model with
    Multi-Layer Response-Isolated Pooling.
    """

    # General Setup
    seed = 42
    debug = False  # Set to True for fast debugging runs with subsets
    n_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    # Metadata files are already generated in ./metadata
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"
    sample_submission_path = "./input/sample_submission.csv"

    # Output Directory
    working_dir = "./working/idea_7/"
    output_dir = os.path.join(working_dir, "output")
    cache_dir = os.path.join(working_dir, "cache")
    submission_path = "./submission/submission.csv"

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Model Architecture
    model_name = "microsoft/deberta-v3-base"
    max_length = 512
    num_classes = 3  # Winner A, Winner B, Tie

    # Specific Architectural Features
    # Multi-Layer Response-Isolated Pooling settings
    pooling_layers = 4  # Number of last hidden layers to aggregate
    use_scalar_features = (
        True  # Whether to use length features (prompt, resp_a, resp_b)
    )

    # Training Hyperparameters
    epochs = 3
    train_batch_size = 8  # Adjusted for A100 40GB with Base model + 512 seq len
    valid_batch_size = 16
    learning_rate = 1e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # Gradient Accumulation
    # Effective batch size = train_batch_size * accumulation_steps
    accumulation_steps = 1

    # Mixed Precision
    use_fp16 = True

    # Inference
    tta = True  # Test Time Augmentation (predict (A,B) and (B,A))

    def __init__(self):
        # Print config on init for logging purposes
        print(f"Config initialized. Device: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Output Directory: {self.working_dir}")
