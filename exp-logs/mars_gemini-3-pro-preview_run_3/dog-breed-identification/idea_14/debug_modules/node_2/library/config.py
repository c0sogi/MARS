import os
import torch


class Config:
    """
    Configuration class for the Dog Breed Prediction task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # Model Architecture
    # ==========================================
    # Using ConvNeXt-Small pre-trained on ImageNet-21k and fine-tuned on 1k
    # as identified in the strategy.
    model_name = "convnext_small.fb_in22k_ft_in1k"

    # Fixed input resolution (224x224) to avoid overfitting and allow stable batch sizes
    image_size = 224

    # Number of breed classes
    num_classes = 120

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size: Safe for A100 40GB with ConvNeXt Small
    batch_size = 64

    # Learning Rate: Conservative rate for fine-tuning to preserve pre-trained features
    lr = 1e-5

    # Scheduler settings (Cosine Annealing)
    min_lr = 1e-7

    # Epochs: 30 epochs for full convergence per fold
    epochs = 30

    # Warmup: 1 epoch to align the head before full fine-tuning
    warmup_epochs = 1

    # Cross-Validation: 5-Fold Stratified
    n_folds = 5

    # Reproducibility
    seed = 42

    # Compute
    # 12 vCPUs available, 4 workers is a safe standard
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data for quick pipeline verification
    debug = False
    debug_subset_size = 100

    # ==========================================
    # Directories and Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Working directory for Idea 14 (Cache and Outputs)
    working_dir = "./working/idea_14"

    # Metadata Paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Sample Submission
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Output Paths
    submission_path = "./submission/submission.csv"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.submission_path), exist_ok=True)


# Execute setup to ensure directories exist when module is imported
Config.setup()
