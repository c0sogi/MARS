import os
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification.
    Centralizes hyperparameters for Model, Data, Training, and Inference.
    """

    # ==========================================
    # System & General
    # ==========================================
    seed = 42
    num_workers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Directories & Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_5"
    submission_dir = "./submission"

    # Metadata Paths
    # Note: For 5-Fold CV, we will likely merge train and val metadata
    # to perform our own stratified splits.
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # ==========================================
    # Data & Augmentation
    # ==========================================
    image_size = 224  # Standard input size for ConvNeXt
    num_classes = 5

    # MixUp / CutMix Params
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 1.0  # Probability to apply mixing (MixUp or CutMix)

    # ==========================================
    # Model Architecture
    # ==========================================
    # Using ConvNeXt Small pre-trained on ImageNet-21k and fine-tuned on 1k
    model_name = "convnext_small.in12k_ft_in1k"
    drop_path_rate = 0.4  # High stochastic depth rate for regularization
    dropout = 0.0  # Head dropout

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    n_folds = 5  # 5-Fold Stratified Cross-Validation
    epochs = 10  # Max epochs per fold
    batch_size = 32  # Batch size of 32 as requested

    # Optimizer (AdamW)
    lr = 1e-4  # Peak learning rate
    min_lr = 1e-6  # Minimum learning rate for Cosine Annealing
    weight_decay = 0.05  # Standard weight decay for ConvNeXt

    # Scheduler
    warmup_epochs = 1

    # ==========================================
    # Inference
    # ==========================================
    tta = True  # Enable Test Time Augmentation (Horizontal Flip)

    # ==========================================
    # Debug / Development
    # ==========================================
    debug = False  # Set to True to run on a small subset for testing
    debug_sample_size = 500

    def __init__(self):
        """
        Initialize configuration and create necessary directories.
        """
        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # Adjust settings if in debug mode
        if self.debug:
            self.epochs = 2
            self.n_folds = 2
