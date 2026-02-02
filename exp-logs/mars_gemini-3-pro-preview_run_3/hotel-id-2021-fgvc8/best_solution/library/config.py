import os
import torch


class Config:
    """
    Configuration class for Hotel Identification Task (Idea 8).
    Implements settings for EfficientNet-V2-S + GeM + Sub-center ArcFace.
    """

    # --------------------
    # General Settings
    # --------------------
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # --------------------
    # Directories
    # --------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_8"

    # Ensure working directory exists immediately upon config load
    os.makedirs(working_dir, exist_ok=True)

    # File Paths
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Output Paths
    model_path = os.path.join(working_dir, "model_v2s_arcface.pth")
    submission_path = "./submission/submission.csv"

    # --------------------
    # Data Parameters
    # --------------------
    # Fixed resolution as per strategy to maximize throughput
    image_size = 224

    # Batch size optimized for A100 40GB with V2-S backbone
    batch_size = 128

    # Number of classes (Hotels)
    num_classes = 7770

    # --------------------
    # Model Architecture
    # --------------------
    # Backbone: EfficientNet-V2-S (timm implementation)
    model_name = "tf_efficientnetv2_s"
    pretrained = True

    # Pooling
    pooling_type = "gem"  # Generalized Mean Pooling

    # Dimensionality of embeddings before the head
    embedding_size = 512

    # --------------------
    # ArcFace / Sub-center Parameters
    # --------------------
    # Scale factor (s)
    scale = 30.0

    # Margin (m)
    margin = 0.5

    # Sub-centers (K) per class for handling intra-class variance
    k_subcenters = 3

    # --------------------
    # Training Hyperparameters
    # --------------------
    epochs = 12

    # Learning Rate (AdamW)
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-2

    # Scheduler
    scheduler_name = "cosine"  # Cosine Annealing
    warmup_epochs = 1  # Linear warmup for stability

    # Early Stopping
    patience = 4

    # --------------------
    # Debug / Development
    # --------------------
    # Set to True to run on a small subset of data for pipeline verification
    debug = False
    debug_sample_size = 1000
