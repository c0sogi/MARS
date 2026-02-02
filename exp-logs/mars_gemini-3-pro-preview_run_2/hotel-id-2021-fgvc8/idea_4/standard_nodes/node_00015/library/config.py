import os
import torch


class CFG:
    """
    Configuration class for Hierarchical Multi-Task Metric Learning model.
    """

    # ====================================================
    # General Settings
    # ====================================================
    seed = 42
    num_workers = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 100

    # ====================================================
    # Directories & Paths
    # ====================================================
    input_dir = "./input"
    train_metadata_path = "./metadata/train_metadata.csv"
    val_metadata_path = "./metadata/val_metadata.csv"
    test_metadata_path = "./metadata/test_metadata.csv"

    # Output directory for checkpoints and cached data
    working_dir = "./working/idea_4"
    os.makedirs(working_dir, exist_ok=True)

    # ====================================================
    # Data Configuration
    # ====================================================
    image_size = 384

    # Debugging / Development
    debug = False
    debug_sample_size = 2000  # Number of samples to use when debug=True

    # ====================================================
    # Model Architecture
    # ====================================================
    backbone = "efficientnet_b5"
    pretrained = True
    embedding_size = 512
    pooling = "gem"  # Generalized Mean Pooling

    # ====================================================
    # Metric Learning Heads
    # ====================================================
    # Head 1: Fine-Grained (Hotel ID) - SubCenter ArcFace
    num_classes = 7770
    subcenter_k = 3
    margin_hotel = 0.2
    scale_hotel = 30.0

    # Head 2: Coarse-Grained (Chain ID) - ArcFace
    num_chains = 88
    margin_chain = 0.5
    scale_chain = 30.0

    # Multi-Task Loss Weight
    lambda_chain = 0.5  # Weight for the chain classification loss

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    epochs = 12
    batch_size = 8  # Tuned for A100 40GB with EfficientNet-B5 @ 384

    # Optimizer & Scheduler
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    scheduler = "CosineAnnealingLR"

    # Label Smoothing (Explicitly Disabled as per requirements)
    label_smoothing = 0.0

    # ====================================================
    # Inference & Post-Processing
    # ====================================================
    # Test-Time Augmentation (Horizontal Flip)
    use_tta = True

    # Query Expansion
    use_qe = True
    qe_k = 3  # Number of neighbors to use for average query expansion
