import os
import torch


class Config:
    """
    Configuration class for Diabetic Retinopathy Severity Prediction.
    Implements settings for 'Idea 9': ConvNeXt-Base with Dual-Stream Pooling (GAP+GeM),
    Ordinal Regression, and Model EMA.
    """

    # ==========================================
    # General Setup
    # ==========================================
    project_name = "diabetic_retinopathy_ordinal"
    idea_name = "idea_9"
    seed = 42
    debug = False  # Set to True to run with a small subset of data for debugging

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of dataloader workers

    # ==========================================
    # Directories & Paths
    # ==========================================
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = os.path.join("./working", idea_name)

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # Metadata Files
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")
    sample_submission_csv = os.path.join(input_dir, "sample_submission.csv")

    # Model Checkpoint Paths
    best_model_path = os.path.join(working_dir, "best_model.pth")
    last_model_path = os.path.join(working_dir, "last_model.pth")
    submission_path = os.path.join(working_dir, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    image_size = 512
    # Batch size optimized for A100 40GB with ConvNeXt-Base + AMP
    batch_size = 8

    # Debugging/Subset Control
    # If not None, these limit the number of samples used during training/validation
    train_subset_size = None
    val_subset_size = None

    # ==========================================
    # Model Architecture
    # ==========================================
    model_name = "convnext_base"
    # Ordinal Regression: 5 classes (0,1,2,3,4) -> 4 binary classification units
    # Unit k predicts prob(y > k)
    num_classes = 4
    pretrained = True

    # Regularization inside model
    drop_rate = 0.0  # Head dropout
    drop_path_rate = 0.2  # Stochastic depth rate (essential for ConvNeXt)

    # Pooling Strategy
    # 'dual' implies concatenation of Global Average Pooling and Generalized Mean Pooling
    pooling_type = "dual"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    epochs = 20
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 0.05  # High weight decay as recommended for ConvNeXt

    # Scheduler
    scheduler = "cosine"  # Cosine Annealing
    warmup_epochs = 1

    # Model EMA (Exponential Moving Average)
    use_ema = True
    ema_decay = 0.999

    # Mixed Precision
    use_amp = True

    # ==========================================
    # Augmentation
    # ==========================================
    # Probability for geometric augmentations (flips, rotations)
    aug_prob = 0.5

    # ==========================================
    # Inference / TTA
    # ==========================================
    # Number of TTA views: 4 (Original, HFlip, VFlip, Rotate180)
    tta_rounds = 4
