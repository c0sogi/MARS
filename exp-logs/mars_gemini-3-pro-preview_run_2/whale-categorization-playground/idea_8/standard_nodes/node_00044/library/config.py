import os
import torch


class CFG:
    """
    Configuration class for the Whale Species Prediction pipeline.
    Implements the High-Fidelity Dual-Ensemble Strategy (Idea 8).
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    seed = 42
    debug = False
    # Use available vCPUs, leaving some overhead for system processes
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_freq = 50

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    input_root = "./input"
    train_dir = os.path.join(input_root, "train")
    test_dir = os.path.join(input_root, "test")

    metadata_root = "./metadata"
    train_csv = os.path.join(metadata_root, "train.csv")
    val_csv = os.path.join(metadata_root, "val.csv")
    test_csv = os.path.join(metadata_root, "test.csv")

    sample_submission = os.path.join(input_root, "sample_submission.csv")

    # Working directory for Idea 8
    working_dir = "./working/idea_8"
    os.makedirs(working_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Normalization constants (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Progressive Resizing Strategy
    image_size_p1 = 256  # Phase 1: Warm-up
    image_size_p2 = 384  # Phase 2: Fine-tuning / High-Fidelity

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Dual-Backbone Ensemble
    model_a_name = "tf_efficientnet_b5_ns"
    model_b_name = "tf_efficientnetv2_m"

    embedding_size = 512
    pool_type = "gem"  # Generalized Mean Pooling

    # Enable Gradient Checkpointing to fit B5/V2-M @ 384px with Batch Size 32
    use_gradient_checkpointing = True

    # -------------------------------------------------------------------------
    # ArcFace Head Hyperparameters
    # -------------------------------------------------------------------------
    arcface_s = 30.0
    arcface_m = 0.50

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Gradient Checkpointing allows batch_size=32 on A100 with these models
    batch_size = 32
    val_batch_size = 64

    learning_rate = 3e-4
    weight_decay = 1e-6

    # Epochs Split: ~60% Phase 1, ~40% Phase 2
    # Total 20 epochs ensures convergence without over-fitting
    epochs_p1 = 12
    epochs_p2 = 8

    # Scheduler Parameters (Cosine Annealing)
    scheduler_params = {
        "T_max": 20,  # epochs_p1 + epochs_p2
        "eta_min": 1e-6,
        "last_epoch": -1,
    }

    # -------------------------------------------------------------------------
    # Inference / Post-Processing
    # -------------------------------------------------------------------------
    # Similarity Threshold for Open-Set Rejection (new_whale)
    # If cosine similarity < threshold, predict new_whale
    inference_threshold = 0.45

    # Test-Time Augmentation
    tta_flips = True  # Average embeddings of image and its horizontal flip
