import os
import torch
import random
import numpy as np


class Config:
    # --- General Settings ---
    seed = 42
    debug = False  # Set to True to train on a small subset for debugging
    debug_sample_size = 100  # Number of samples to use when debug=True
    num_workers = 4

    # --- Directories ---
    # Input directories (Read-Only)
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_images_dir = os.path.join(input_dir, "train_images")
    test_images_dir = os.path.join(input_dir, "test_images")

    # Metadata paths
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Output directories (Write Allowed)
    working_dir = "./working/idea_3"
    submission_dir = "./submission"

    # Output file paths
    model_save_path = os.path.join(working_dir, "best_model.pth")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # --- Data Configuration ---
    # Class labels sorted alphabetically (consistent with MultiLabelBinarizer)
    class_labels = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]
    num_classes = len(class_labels)

    # --- Model Architecture ---
    # Using EfficientNetV2 Medium pre-trained on ImageNet-21k and fine-tuned on 1k
    # Scaling up resolution to 384x384 for fine-grained feature detection.
    model_name = "tf_efficientnetv2_m.in21k_ft_in1k"
    img_size = 384

    # --- Training Hyperparameters ---
    epochs = 15
    batch_size = 24  # Adjusted for larger model and resolution

    # Optimization
    learning_rate = 1e-4
    min_lr = 1e-6
    weight_decay = 0.05
    warmup_epochs = 1

    # Loss
    label_smoothing = 0.05

    # --- Inference ---
    threshold = 0.5  # Threshold for multi-label classification

    # --- Compute ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Set seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.seed)
            torch.cuda.manual_seed_all(cls.seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Config setup complete. Output directory: {cls.working_dir}")
        print(f"Device: {cls.device}")
