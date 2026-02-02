import os
import torch


class Config:
    """
    Configuration class for the Triple Heterogeneous Ensemble strategy.
    Defines hyperparameters, paths, and model architectures.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs for data loading
    NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 4

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Metadata CSVs (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size adjusted for A100 GPU memory (40GB) with these backbones
    BATCH_SIZE = 32

    # Training duration
    EPOCHS = 10

    # Optimization (AdamW + Cosine Annealing)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6
    T_MAX = EPOCHS  # Cycle length for Cosine Annealing

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500

    # -------------------------------------------------------------------------
    # Model Architectures (Triple Heterogeneous Ensemble)
    # -------------------------------------------------------------------------
    # Defines the specific backbones, resolutions, and structural modifications.
    MODEL_ARCHITECTURES = [
        # 1. ResNet-50 (Standard CNN)
        # Modification: GeM Pooling for salient feature focus.
        # Resolution: 256x256 for fine spatial details.
        {
            "model_name": "resnet50.a1_in1k",
            "resolution": 256,
            "use_gem": True,
            "use_msd": False,
        },
        # 2. ConvNeXt-Small (Modern CNN)
        # Modification: GeM Pooling.
        # Resolution: 256x256.
        {
            "model_name": "convnext_small.fb_in1k",
            "resolution": 256,
            "use_gem": True,
            "use_msd": False,
        },
        # 3. MaxViT-Tiny (Multi-Axis Transformer)
        # Modification: Multi-Sample Dropout (MSD) for Log Loss optimization.
        # Resolution: 224x224 (Native grid alignment).
        {
            "model_name": "maxvit_tiny_tf_224.in1k",
            "resolution": 224,
            "use_gem": False,
            "use_msd": True,
        },
    ]

    @classmethod
    def setup(cls):
        """
        Ensures necessary working and output directories exist.
        Should be called at the start of the execution pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
