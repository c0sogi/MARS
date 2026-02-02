import os
import torch


class Config:
    """
    Configuration class for the Apple Disease Detection task.
    Centralizes hyperparameters, paths, and model settings.
    """

    # =======================
    # General Settings
    # =======================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =======================
    # Directories & Paths
    # =======================
    input_dir = "./input"
    train_images_dir = os.path.join(input_dir, "train_images")
    test_images_dir = os.path.join(input_dir, "test_images")

    # Metadata paths (pre-generated)
    train_meta_path = "./metadata/train.csv"
    val_meta_path = "./metadata/val.csv"
    test_meta_path = "./metadata/test.csv"
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Working directory for Idea 4
    work_dir = "./working/idea_4"
    os.makedirs(work_dir, exist_ok=True)

    # Output paths
    submission_path = os.path.join(work_dir, "submission.csv")

    # =======================
    # Data Configuration
    # =======================
    img_size = 512
    num_classes = 6
    # Class labels sorted alphabetically to ensure consistent mapping
    class_labels = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # =======================
    # Model Architecture
    # =======================
    # List of models for the ensemble
    model_names = [
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        "maxvit_base_tf_512.in1k",
    ]

    # =======================
    # Training Hyperparameters
    # =======================
    epochs = 20

    # Batch size configuration
    # 512x512 resolution is memory intensive.
    # We use a small physical batch size and gradient accumulation to achieve a stable effective batch size.
    batch_size = 8
    gradient_accumulation_steps = 4  # Effective batch size = 8 * 4 = 32

    # Optimizer settings
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    max_grad_norm = 1.0

    # Scheduler settings
    warmup_epochs = 1

    # =======================
    # Loss Function (Asymmetric Loss)
    # =======================
    # Parameters to handle class imbalance and multi-label nature
    asl_gamma_neg = 4.0
    asl_gamma_pos = 0.0
    asl_clip = 0.05

    # =======================
    # Inference
    # =======================
    use_tta = True  # Use Test Time Augmentation (Horizontal Flip)
