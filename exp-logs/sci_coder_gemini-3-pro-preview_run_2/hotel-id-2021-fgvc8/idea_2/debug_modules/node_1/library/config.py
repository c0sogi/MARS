import os
import torch


class Config:
    """
    Configuration class for Hotel ID Deep Metric Learning task.
    Centralizes all hyperparameters, paths, and settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # Number of dataloader workers
    print_freq = 100  # Frequency of logging during training

    # =========================================================================
    # Data Paths
    # =========================================================================
    input_dir = "./input"
    train_images_dir = os.path.join(input_dir, "train_images")
    test_images_dir = os.path.join(input_dir, "test_images")

    metadata_dir = "./metadata"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # =========================================================================
    # Output Paths
    # =========================================================================
    working_dir = "./working/idea_2"
    submission_dir = "./submission"

    model_save_path = os.path.join(working_dir, "model.pth")
    # Cache file for generated embeddings during inference
    gallery_embeddings_path = os.path.join(working_dir, "gallery_embeddings.parquet")
    query_embeddings_path = os.path.join(working_dir, "query_embeddings.parquet")

    submission_path = os.path.join(submission_dir, "submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    backbone_name = "resnet50"
    embedding_size = 512
    num_classes = 7770  # As identified in data analysis
    pretrained = True

    # =========================================================================
    # ArcFace Hyperparameters
    # =========================================================================
    s = 30.0  # Scale factor
    m = 0.50  # Angular margin
    easy_margin = False
    ls_eps = 0.0  # Label smoothing epsilon

    # =========================================================================
    # Image Preprocessing / Augmentation
    # =========================================================================
    img_size = 256  # Resize dimension
    crop_size = 224  # Input dimension to the model

    # ImageNet Normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    epochs = 12
    batch_size = 64  # Suitable for A100 with ResNet50
    lr = 1e-4  # Learning rate for backbone
    weight_decay = 1e-2  # Weight decay for AdamW

    # Scheduler settings (Cosine Annealing)
    min_lr = 1e-6
    warmup_epochs = 1

    # =========================================================================
    # Inference / Retrieval Hyperparameters
    # =========================================================================
    knn = 100  # Number of neighbors to retrieve
    top_k = 5  # Number of predictions to submit per image

    def __init__(self):
        """
        Initialize configuration and create necessary directories.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
