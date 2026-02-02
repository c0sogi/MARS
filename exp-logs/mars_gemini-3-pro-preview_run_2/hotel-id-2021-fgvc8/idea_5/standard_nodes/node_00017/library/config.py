import os
import torch


class Config:
    # =========================================================================
    # General System Settings
    # =========================================================================
    seed = 42
    num_workers = 8  # Utilizing available vCPUs (12 available, leaving buffer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_5"

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Metadata Paths
    train_metadata_path = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata.csv")

    # Model Checkpoint Paths
    model_save_path = os.path.join(working_dir, "model.pth")
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # Cache / Embeddings Paths
    gallery_embeddings_path = os.path.join(working_dir, "gallery_embeddings.parquet")
    query_embeddings_path = os.path.join(working_dir, "query_embeddings.parquet")

    # Submission Path
    submission_path = os.path.join(working_dir, "submission.csv")

    # =========================================================================
    # Model Hyperparameters (Idea 5)
    # =========================================================================
    # Backbone: ConvNeXt-Base initialized with ImageNet-21k weights
    backbone = "convnext_base.fb_in22k"
    embedding_size = 512
    num_classes = 7770  # Based on dataset analysis

    # Sub-Center ArcFace Head
    margin = 0.50
    scale = 30.0
    k_subcenters = 3  # K=3 sub-centers per class to handle multi-modality

    # =========================================================================
    # Data Configuration
    # =========================================================================
    image_size = 384  # Resolution 384x384

    # Normalization (ImageNet defaults)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Configuration
    # =========================================================================
    epochs = 15

    # Class-Balanced (P-K) Sampling
    classes_per_batch = 4  # P
    samples_per_class = 4  # K
    batch_size = classes_per_batch * samples_per_class  # 16

    # Optimizer (AdamW) & Scheduler (Cosine Annealing)
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-2
    scheduler_T_max = epochs  # For CosineAnnealingLR

    # =========================================================================
    # Inference / Post-Processing (Idea 5)
    # =========================================================================
    # Database Augmentation (DBA)
    use_dba = True
    dba_neighbors = 5  # Number of neighbors to aggregate for DBA

    # Query Expansion (QE)
    use_qe = True

    # =========================================================================
    # Debugging
    # =========================================================================
    debug = False  # Set to True to run on a small subset
    debug_sample_size = 1000  # Number of images to use in debug mode
