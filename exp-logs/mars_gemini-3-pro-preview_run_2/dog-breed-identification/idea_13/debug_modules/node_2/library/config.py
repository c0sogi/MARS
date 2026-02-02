import os


class Config:
    """
    Configuration class for the Hybrid Supervised-SSL Multi-View Ensemble strategy.
    Defines paths, hyperparameters, and model specifications.
    """

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Metadata Paths
    # -------------------------------------------------------------------------
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # General Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda"  # Assumes NVIDIA A100 is available

    # -------------------------------------------------------------------------
    # Model Architectures
    # -------------------------------------------------------------------------
    # Stream A: Supervised CNN (ConvNeXt Large)
    # Using torchvision "New Recipe" equivalent in timm or specific tag
    MODEL_A_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Stream B: Self-Supervised ViT (DINOv2 Large)
    MODEL_B_NAME = "vit_large_patch14_dinov2.lvd142m"

    # -------------------------------------------------------------------------
    # Multi-View Configuration
    # -------------------------------------------------------------------------
    # Defines the geometric transformations for the three views.
    # 'resize': int (maintain aspect ratio) or tuple (squish)
    # 'crop': int (center crop size)

    VIEW_GLOBAL = {
        "name": "global",
        "resize": (224, 224),  # Squish to 224x224
        "crop": None,
    }

    VIEW_STANDARD = {
        "name": "standard",
        "resize": 232,  # Resize short edge to 232
        "crop": 224,  # Center crop 224
    }

    VIEW_LOCAL = {
        "name": "local",
        "resize": 288,  # Resize short edge to 288 (Zoom)
        "crop": 224,  # Center crop 224
    }

    VIEWS = [VIEW_GLOBAL, VIEW_STANDARD, VIEW_LOCAL]

    # -------------------------------------------------------------------------
    # Caching Paths
    # -------------------------------------------------------------------------
    # To prevent re-computation of embeddings and features

    # Stream A Caches
    STREAM_A_TRAIN_EMB = os.path.join(WORKING_DIR, "stream_a_train_emb.npy")
    STREAM_A_VAL_EMB = os.path.join(WORKING_DIR, "stream_a_val_emb.npy")
    STREAM_A_TEST_EMB = os.path.join(WORKING_DIR, "stream_a_test_emb.npy")
    STREAM_A_MODEL = os.path.join(WORKING_DIR, "stream_a_logreg.joblib")

    # Stream B Caches
    STREAM_B_TRAIN_EMB = os.path.join(WORKING_DIR, "stream_b_train_emb.npy")
    STREAM_B_VAL_EMB = os.path.join(WORKING_DIR, "stream_b_val_emb.npy")
    STREAM_B_TEST_EMB = os.path.join(WORKING_DIR, "stream_b_test_emb.npy")
    STREAM_B_MODEL = os.path.join(WORKING_DIR, "stream_b_logreg.joblib")

    # Shared Label/ID Caches
    TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_LABELS_CACHE = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_IDS_CACHE = os.path.join(WORKING_DIR, "test_ids.npy")

    # -------------------------------------------------------------------------
    # Classifier Hyperparameters
    # -------------------------------------------------------------------------
    # LogisticRegressionCV parameters
    LOGREG_PARAMS = {
        "Cs": 10,  # Number of C values to try in grid search
        "cv": 5,  # Cross-validation folds
        "max_iter": 2000,  # Max iterations for convergence
        "solver": "lbfgs",  # Good for multiclass
        "multi_class": "multinomial",
        "n_jobs": -1,  # Use all CPUs
        "random_state": SEED,
    }
