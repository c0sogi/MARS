import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection Task.
    Implements settings for 'Feature Pyramid Ensemble with Deep Supervision'.
    """

    # ==== General Settings ====
    project_name = "apple_disease_detection"
    idea_name = "idea_12"
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging

    # ==== Compute ====
    # Use all available CPUs for data loading
    num_workers = os.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Automatic Mixed Precision
    use_amp = True

    # ==== Directories ====
    input_root = "./input"
    metadata_root = "./metadata"
    # Specific working directory for this idea
    working_dir = os.path.join("./working", idea_name)

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Data Paths (using pre-generated metadata)
    train_csv_path = os.path.join(metadata_root, "train.csv")
    val_csv_path = os.path.join(metadata_root, "val.csv")
    test_csv_path = os.path.join(metadata_root, "test.csv")
    image_dir = os.path.join(input_root, "images")

    # Cache Directory for processed datasets (parquet/npy)
    cache_dir = working_dir

    # ==== Task Configuration ====
    num_classes = 4
    class_labels = ["healthy", "multiple_diseases", "rust", "scab"]

    # ==== Model Architectures ====
    # Heterogeneous Ensemble of Fused-Pyramid Experts
    models = [
        {
            "name": "tf_efficientnetv2_m.in21k_ft_in1k",
            "img_size": 512,
            "batch_size": 16,  # Optimized for A100 40GB
            "dropout": 0.3,
            "drop_path": 0.2,
            "feature_dim": 1280,  # Last channel dim for FPN
        },
        {
            "name": "maxvit_small_tf_384.in1k_ft_in1k",
            "img_size": 384,
            "batch_size": 16,  # Optimized for A100 40GB
            "dropout": 0.0,  # MaxViT typically requires less explicit dropout
            "drop_path": 0.2,
            "feature_dim": 768,  # Last channel dim for FPN
        },
    ]

    # ==== Training Hyperparameters ====
    epochs = 25
    # Relaxed patience for EMA convergence
    patience = 10

    # Optimization
    learning_rate = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    max_grad_norm = 10.0

    # Loss Settings
    # Deep Supervision: Total = L_final + 0.5 * (L_aux1 + L_aux2)
    aux_loss_weight = 0.5
    use_class_weights = True  # Use Inverse Frequency weights

    # EMA Settings
    model_ema = True
    model_ema_decay = 0.999

    # ==== Augmentation ====
    # Strong Geometric Augmentation (ShiftScaleRotate, RandomFlip)
    # No Cutout/CoarseDropout (preserves small lesions)
    # No Brightness/Contrast (preserves color signals)
    aug_prob = 0.5

    # ==== Inference / TTA ====
    # Domain-Aware TTA: Original + Horizontal Flip only
    # Vertical flips/Transpose excluded due to gravity priors
    tta_transforms = ["horizontal"]

    # ==== Submission ====
    submission_path = os.path.join(working_dir, "submission.csv")

    @classmethod
    def print_config(cls):
        """Prints the configuration settings."""
        print(f"\n{'='*20} CONFIGURATION {'='*20}")
        print(f"Device: {cls.device}")
        print(f"Working Dir: {cls.working_dir}")
        print(f"Models: {[m['name'] for m in cls.models]}")
        print(f"Epochs: {cls.epochs}, Patience: {cls.patience}")
        print(f"Batch Sizes: {[m['batch_size'] for m in cls.models]}")
        print(f"{'='*55}\n")
