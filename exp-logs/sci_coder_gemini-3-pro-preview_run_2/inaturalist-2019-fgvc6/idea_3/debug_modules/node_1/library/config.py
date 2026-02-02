import os
import torch


class CFG:
    """
    Configuration class containing all hyperparameters and settings for the project.
    """

    # ================= General Settings =================
    seed = 42
    debug = False  # Set to True to run on a small subset of data
    debug_sample_size = 1000  # Number of samples to use when debug is True

    # ================= Data Information =================
    num_classes = 1010
    input_root = "./input"
    train_metadata_path = "./metadata/train.csv"
    val_metadata_path = "./metadata/val.csv"
    test_metadata_path = "./metadata/test.csv"

    # ================= Model Architecture =================
    # Using EfficientNetV2-Small as per the hardware-aware strategy
    model_name = "tf_efficientnetv2_s"
    pretrained = True
    drop_path_rate = 0.2  # Stochastic depth regularization

    # ================= Input Processing =================
    # Resolution Discrepancy Strategy: Train low, Test high
    train_size = 256
    test_size = 384

    # ================= DataLoader =================
    batch_size = 64
    num_workers = 8  # Utilizing available vCPUs

    # ================= Training Hyperparameters =================
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-2
    label_smoothing = 0.1

    # ================= Scheduler =================
    scheduler_type = "CosineAnnealingLR"
    min_lr = 1e-6

    # ================= Hardware =================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================= Output Directories =================
    output_dir = "./working/idea_3"
    submission_dir = "./submission"

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary output directories exist.
        """
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)


# Automatically create directories when imported/used
CFG.setup()
