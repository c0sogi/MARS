import os
import torch


class Config:
    """
    Configuration for Animal Classification Task (Idea 3).
    Implements strategy: ConvNeXt-Tiny + Full Dataset + Mixup/CutMix + Class-Balanced Focal Loss + EMA.
    """

    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    EMA_MODEL_PATH = os.path.join(WORKING_DIR, "ema_model.pth")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 23
    NUM_WORKERS = 12
    BATCH_SIZE = 128  # A100 40GB can handle this easily for Tiny model

    # ==========================================
    # Model Architecture
    # ==========================================
    # Using ConvNeXt Tiny pre-trained on ImageNet-21k and fine-tuned on 1k
    MODEL_NAME = "convnext_tiny.in12k_ft_in1k"
    PRETRAINED = True
    DROP_PATH_RATE = 0.1  # Stochastic depth rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 20  # Increased epochs as Mixup requires longer training

    # Optimizer (AdamW)
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 2
    MIN_LR = 1e-6

    # Gradient Clipping
    CLIP_GRAD = 5.0

    # Precision
    USE_AMP = True  # Automatic Mixed Precision

    # ==========================================
    # Regularization & Imbalance Handling
    # ==========================================
    # Strategy: No Sampler, Use Mixup/CutMix + Class Balanced Loss
    USE_WEIGHTED_SAMPLER = False

    # Mixup / CutMix Params
    USE_MIXUP_CUTMIX = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0  # Probability of applying mixup or cutmix
    MIXUP_SWITCH_PROB = 0.5  # Probability of switching to cutmix instead of mixup
    MIXUP_MODE = "batch"  # Apply same mixup params across batch

    # Class Balanced Focal Loss Params
    LOSS_TYPE = "class_balanced_focal"
    FOCAL_GAMMA = 2.0
    CLASS_BETA = 0.9999  # Beta for calculating effective number of samples

    # Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.9999
    EMA_UPDATE_EVERY = 1

    # ==========================================
    # Class Mapping
    # ==========================================
    LABEL_MAP = {
        0: "empty",
        1: "deer",
        2: "moose",
        3: "squirrel",
        4: "rodent",
        5: "small_mammal",
        6: "elk",
        7: "pronghorn_antelope",
        8: "rabbit",
        9: "bighorn_sheep",
        10: "fox",
        11: "coyote",
        12: "black_bear",
        13: "raccoon",
        14: "skunk",
        15: "wolf",
        16: "bobcat",
        17: "cat",
        18: "dog",
        19: "opossum",
        20: "bison",
        21: "mountain_goat",
        22: "mountain_lion",
    }

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_device(cls):
        """Returns the computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
