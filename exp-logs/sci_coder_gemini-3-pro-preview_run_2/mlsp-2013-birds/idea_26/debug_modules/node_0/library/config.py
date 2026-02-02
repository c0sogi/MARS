import os
import torch


class CFG:
    """
    Configuration class for Heterogeneous Ensemble with Class-Weighted Knowledge Distillation.
    """

    # General
    seed = 42
    debug = False
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Working Directory (Cache & Outputs)
    working_dir = "./working/idea_26"

    # Data Parameters
    # Aspect ratio 1:2 (Freq x Time) to preserve temporal fidelity
    img_height = 224
    img_width = 448
    num_classes = 19

    # Training Parameters
    n_folds = 5
    epochs = 50
    batch_size = 16  # Stable batch size for small dataset

    # Optimization
    lr = 1e-3
    weight_decay = 1e-4

    # Model Architecture
    # Anchors: Stable baselines
    model_anchors = ["resnet18", "efficientnet_b0"]
    # Student: High capacity, prone to instability without guidance
    model_student = "densenet121"
    pretrained = True

    # Regularization & Augmentation
    mixup_alpha = 0.4

    # Distillation
    distillation_lambda = 1.0

    # Class Imbalance Handling (BCEWithLogitsLoss pos_weight)
    # Calculated from EDA: Total Training Samples = 206
    # Counts: [6, 22, 16, 3, 7, 4, 15, 15, 19, 17, 39, 8, 7, 3, 13, 5, 2, 3, 10]
    _class_counts = [6, 22, 16, 3, 7, 4, 15, 15, 19, 17, 39, 8, 7, 3, 13, 5, 2, 3, 10]
    _total_samples = 206

    # Formula: pos_weight = (Total - Count) / Count
    # This balances the loss contribution of positive vs negative examples for each class
    pos_weights = [(_total_samples - c) / c for c in _class_counts]


# Ensure the working directory exists as per requirements
os.makedirs(CFG.working_dir, exist_ok=True)
