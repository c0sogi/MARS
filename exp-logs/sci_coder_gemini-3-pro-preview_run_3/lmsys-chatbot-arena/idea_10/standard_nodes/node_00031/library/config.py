import os
import torch


class Config:
    # ==== General Settings ====
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    exp_name = "idea_10"

    # ==== Paths ====
    # Working directory for artifacts
    working_dir = os.path.join("./working", exp_name)

    # Cache directory for processed data (parquet/numpy)
    cache_dir = os.path.join(working_dir, "cache")

    # Output directory for logs and checkpoints
    output_dir = os.path.join(working_dir, "output")

    # Path to save the best model
    model_path = os.path.join(working_dir, "best_model.pth")

    # Submission directory and file
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Input Metadata Paths (Pre-split)
    train_path = "./metadata/train.csv"
    val_path = "./metadata/val.csv"
    test_path = "./metadata/test.csv"

    # ==== Model Architecture ====
    # Using DeBERTa-v3-Large as the backbone for better reasoning
    model_name = "microsoft/deberta-v3-large"

    # Max length per branch (Siamese network processes two branches)
    # Total effective context covers Prompt + Response A and Prompt + Response B
    max_len = 512

    # Classification head settings
    num_classes = 3  # Winner A, Winner B, Tie
    dropout = 0.1

    # ==== Training Hyperparameters ====
    epochs = (
        3  # Large models converge faster; 3 epochs on full data is usually sufficient
    )
    learning_rate = 5e-6  # Conservative LR for Large backbone
    weight_decay = 0.01

    # Batch Size Strategy for A100 40GB
    # DeBERTa-Large is memory intensive. We use a small physical batch size
    # and accumulate gradients to reach a stable target batch size.
    physical_batch_size = 2
    target_batch_size = 64
    gradient_accumulation_steps = int(target_batch_size / physical_batch_size)

    # Optimization
    fp16 = True  # Mixed precision training
    gradient_checkpointing = True  # Save memory at the cost of small compute overhead
    max_grad_norm = 1.0
    patience = 2  # Early stopping patience

    # ==== Data Augmentation ====
    # Symmetric augmentation: Train on (A, B) -> Label and (B, A) -> 1-Label
    augment_symmetric = True

    # ==== Inference ====
    # Test-Time Augmentation: Predict (A, B) and (B, A) and average
    tta = True

    # ==== Hardware ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers

    @classmethod
    def setup(cls):
        """
        Create necessary directories for the experiment.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Print configuration summary
        print(f"Experiment: {cls.exp_name}")
        print(f"Model: {cls.model_name}")
        print(f"Device: {cls.device}")
        print(
            f"Batch Size: {cls.physical_batch_size} (Physical) * {cls.gradient_accumulation_steps} (Accum) = {cls.target_batch_size} (Target)"
        )
