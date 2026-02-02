import os
import torch


class CFG:
    """
    Configuration class for the Heterogeneous Cross-Validation Ensemble strategy.
    Centralizes all hyperparameters for data, models, training, and inference.
    """

    # =======================
    # General Settings
    # =======================
    seed = 42
    debug = False  # Set to True to run on a small subset of data for debugging
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =======================
    # Data Configuration
    # =======================
    # Resolution: 384x384 to capture fine details (fur, whiskers)
    image_size = 384

    # Batch Size: 32 fits A100 40GB for Base/Medium models and maintains BN stability (>16)
    batch_size = 32

    # Cross-Validation: 5 Folds to utilize 100% of data for training across the ensemble
    n_fold = 5

    # Paths
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output directories
    output_dir = "./working/idea_3"
    submission_dir = "./submission"

    # =======================
    # Model Architecture
    # =======================
    # Heterogeneous Ensemble:
    # 1. EfficientNetV2-M (MBConv-based, high efficiency)
    # 2. ConvNeXt-Base (Modernized ResNet, isotropic, ViT-like)
    model_names = ["tf_efficientnetv2_m.in21k_ft_in1k", "convnext_base.fb_in1k"]

    # =======================
    # Training Hyperparameters
    # =======================
    # Duration: Short duration (5 epochs) leveraging pre-trained weights
    epochs = 5

    # Optimization
    learning_rate = 1e-4
    weight_decay = 1e-6
    min_lr = 1e-6  # For Cosine Annealing

    # Early Stopping
    patience = 3

    # Precision
    use_amp = True  # Automatic Mixed Precision

    # =======================
    # Augmentation
    # =======================
    # Heavy Augmentation Strategy (No Mixup/CutMix)
    train_aug_params = {
        "resize_crop_min_scale": 0.8,
        "shift_scale_rotate_prob": 0.5,
        "color_jitter_prob": 0.5,
        "color_jitter_brightness": 0.2,
        "color_jitter_contrast": 0.2,
        "color_jitter_saturation": 0.2,
        "color_jitter_hue": 0.1,
    }

    # =======================
    # Inference
    # =======================
    # Test Time Augmentation: Horizontal Flip
    tta = True
