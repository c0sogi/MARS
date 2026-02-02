import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Strict determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    """
    Configuration class for the Cassava Leaf Disease Classification task.
    Implements the Independent Heterogeneous Ensemble strategy (Idea 7).
    """

    # Meta
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # Compute Environment
    num_workers = 4  # Leveraging available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Paths
    input_root = "./input"
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output Paths
    output_dir = "./working/optimized_vit"
    submission_dir = "./submission"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # Model Architecture
    model_name = "vit_base_patch16_384"

    img_size = 384
    num_classes = 5

    # Training Hyperparameters
    epochs = 10
    # Batch size optimized for A100 40GB memory with 384px resolution
    batch_size = 32

    # Optimizer & Scheduler
    lr = 2e-5
    min_lr = 1e-6
    weight_decay = 1e-4
    optimizer = "AdamW"
    scheduler = "CosineAnnealingLR"
    T_max = epochs  # Cycle length for Cosine Annealing

    # Regularization
    label_smoothing = 0.1
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5  # Probability to apply MixUp/CutMix

    # Validation & Stopping
    patience = 3  # Early stopping patience

    # Inference
    tta_steps = 3  # Number of TTA views (Original + Flips/Transpose)

    # Initialization logic
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
