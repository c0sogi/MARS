import os
import torch


class CFG:
    """
    Configuration class for Apple Disease Detection.
    Encapsulates parameters for Data, Model, Training, and Inference.
    """

    # ==========================
    # General & Compute
    # ==========================
    debug = False  # Set to True for quick debugging with subset
    debug_sample_size = 100  # Number of samples to use when debug=True
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # Directories & Paths
    # ==========================
    input_dir = "./input"
    metadata_dir = "./metadata"
    images_dir = os.path.join(input_dir, "images")

    # Metadata CSVs
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output Paths
    submission_dir = "./submission"
    output_dir = "./working/idea_4"

    # Ensure working directories exist
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ==========================
    # Data Configuration
    # ==========================
    class_labels = ["healthy", "multiple_diseases", "rust", "scab"]
    num_classes = len(class_labels)
    target_col = "stratify_label"  # Used for stratification

    # ==========================
    # Model Architecture
    # ==========================
    # Ensemble Backbones
    backbones = ["tf_efficientnet_b4_ns", "convnext_tiny.fb_in1k"]

    # Input resolutions (EfficientNet-B4: 380, ConvNeXt-Tiny: 224)
    img_sizes = {"tf_efficientnet_b4_ns": 380, "convnext_tiny.fb_in1k": 224}

    # Generalized Mean (GeM) Pooling Config
    use_gem = True
    gem_p = 3.0  # Initial p value
    gem_learnable = True  # Learn p during training

    # ==========================
    # Training Hyperparameters
    # ==========================
    n_folds = 4
    epochs = 15
    batch_size = 32  # Fits within 40GB VRAM for these models

    # Optimizer (AdamW)
    learning_rate = 1e-4
    weight_decay = 1e-4
    min_lr = 1e-6

    # Scheduler (Cosine Annealing)
    T_max = epochs

    # Loss Function
    use_class_weights = True  # Inverse frequency weights for imbalance

    # Early Stopping
    patience = 4

    # ==========================
    # Augmentation & Inference
    # ==========================
    aug_prob = 0.5
    tta_steps = 3  # Original + Horizontal Flip + Vertical Flip
