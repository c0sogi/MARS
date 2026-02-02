import os
import torch


class CFG:
    """
    Configuration class for Cassava Leaf Disease Classification.
    """

    # =====================
    # Meta Configuration
    # =====================
    seed = 42
    debug = False
    debug_sample_size = 1000  # Number of samples to use when debug=True
    num_workers = 8  # Cite solution_lesson_node_00038
    print_freq = 100  # Logging frequency

    # =====================
    # Data Paths
    # =====================
    input_root = "./input"
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"
    test_csv = "./metadata/test.csv"
    output_dir = "./working/idea_2"

    # =====================
    # Model Configuration
    # =====================
    model_name = "tf_efficientnet_b4_ns"
    img_size = 380  # Resolution for EfficientNet-B4
    num_classes = 5
    target_col = "label"

    # =====================
    # Training Hyperparameters
    # =====================
    epochs = 10
    train_batch_size = 32  # Tuned for A100 40GB
    valid_batch_size = 64

    # Optimizer
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6

    # Scheduler (Cosine Annealing Warm Restarts)
    scheduler = "CosineAnnealingWarmRestarts"
    T_0 = 10  # Number of iterations for the first restart
    T_mult = 1  # A factor increases T_i after a restart

    # =====================
    # Regularization & Augmentation
    # =====================
    label_smoothing = 0.1
    mixup_prob = 0.5  # Probability of applying MixUp
    cutmix_prob = 0.5  # Probability of applying CutMix

    # =====================
    # Inference
    # =====================
    tta_steps = 3  # Test Time Augmentation steps (e.g., Original, HFlip, VFlip)

    # =====================
    # System
    # =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Setup the environment, such as creating necessary directories.
        """
        os.makedirs(cls.output_dir, exist_ok=True)


# Initialize environment setup on import
CFG.setup()
