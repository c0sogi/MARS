import os
import torch


class Config:
    """
    Centralized configuration for the Insult Detection pipeline.
    Encapsulates model architecture, training hyperparameters, file paths,
    and feature engineering settings for the Hybrid DeBERTa-v3 solution.
    """

    # =========================================================================
    # General & System Settings
    # =========================================================================
    seed = 42
    debug = False  # Set to True to run on a small subset for rapid debugging
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Metadata (Read-Only)
    metadata_dir = "./metadata"
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "validation.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    # Working Directory (For Cache & Intermediate Outputs)
    working_dir = "./working/idea_8"
    os.makedirs(working_dir, exist_ok=True)

    # Submission Directory
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")
    os.makedirs(submission_dir, exist_ok=True)

    # Cached Structural Feature Paths (Dense SVD vectors stored as .npy)
    train_struct_features_path = os.path.join(working_dir, "train_struct_features.npy")
    val_struct_features_path = os.path.join(working_dir, "val_struct_features.npy")
    test_struct_features_path = os.path.join(working_dir, "test_struct_features.npy")

    # Distillation / Pseudo-Labeling Paths
    # Stores the soft probabilities from the Teacher ensemble
    teacher_soft_labels_path = os.path.join(working_dir, "teacher_soft_labels.npy")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_name = "microsoft/deberta-v3-base"
    max_len = 128  # Maximum sequence length for tokenization

    # Structural Branch Settings
    svd_output_dim = 256  # Dimension of the compressed TF-IDF features

    # Classification Head Settings
    # Variable-Rate Multi-Sample Dropout (VR-MSD) rates
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # TF-IDF N-gram ranges for Structural Branch
    tfidf_word_ngram_range = (1, 2)
    tfidf_char_ngram_range = (3, 5)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    n_folds = 5
    epochs = 4  # Number of training epochs per fold
    batch_size = 16

    # Optimizer Settings
    lr_backbone = 2e-5  # Learning rate for DeBERTa layers
    lr_head = 1e-3  # Learning rate for Custom Head (Fusion + Classifier)
    weight_decay = 0.01
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1

    # Scheduler
    scheduler_type = "cosine"
    warmup_ratio = 0.1

    # =========================================================================
    # Advanced Optimization (AWP)
    # =========================================================================
    use_awp = True
    awp_lr = 1e-4
    awp_eps = 1e-2
    awp_start_epoch = 1  # Epoch index to start AWP (e.g., start after 1st epoch)

    # =========================================================================
    # Distillation Strategy
    # =========================================================================
    # 'teacher': Train initial ensemble on labeled data
    # 'student': Train distilled ensemble on labeled + soft-labeled test data
    inference_mode = "student"
