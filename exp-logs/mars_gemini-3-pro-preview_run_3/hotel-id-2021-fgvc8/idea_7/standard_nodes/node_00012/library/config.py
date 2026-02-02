import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Benchmark = True is faster for fixed input sizes (224x224)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set to True to run on a small subset of data for debugging

    # Directories
    input_root = "./input"
    output_dir = "./working/idea_7/"

    # Metadata Paths
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    image_size = 224  # Resolution set to 224 for high throughput
    n_classes = 7770  # Total unique hotels in the training set
    num_workers = 4  # Number of DataLoader workers
    batch_size = 128  # Batch size optimized for A100 and ConvNeXt-Tiny

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    backbone = "convnext_tiny"  # Structural innovation over EfficientNet
    embedding_size = 512  # Dimension of the embedding vector
    pretrained = True  # Use ImageNet pretrained weights

    # Neck & Head
    use_gem = True  # Generalized Mean Pooling
    use_bn_neck = True  # Batch Normalization Neck
    head = "arcface"  # ArcMarginProduct

    # ArcFace Hyperparameters
    arcface_s = 30.0  # Scale
    arcface_m = 0.50  # Margin
    arcface_ls_eps = 0.0  # Label smoothing epsilon (optional)

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    epochs = 15  # Sufficient for convergence with 70k images
    lr = 3e-4  # Learning rate for AdamW
    min_lr = 1e-6  # Minimum learning rate for Cosine Scheduler
    weight_decay = 0.05  # Weight decay for AdamW
    warmup_epochs = 1  # Linear warmup epochs
    scheduler = "cosine"  # Cosine Annealing scheduler

    # Gradient Clipping
    max_grad_norm = 10.0

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    top_k = 5  # Number of predictions per image (MAP@5)
    tta = True  # Use Test Time Augmentation (Horizontal Flip)

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates the output directory if it does not exist.
        """
        os.makedirs(cls.output_dir, exist_ok=True)


# Initialize environment
Config.setup()
seed_everything(Config.seed)
