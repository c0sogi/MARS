import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Image Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    BACKBONE = "tf_efficientnetv2_s.in1k"
    PRETRAINED = True
    IMG_SIZE = (768, 768)
    IN_CHANNELS = 3

    # Heads
    NUM_CLASSES = 1  # Primary task: Cancer (Binary)
    NUM_BIRADS_CLASSES = 3  # Aux task: BIRADS (0, 1, 2)
    NUM_DENSITY_CLASSES = 4  # Aux task: Density (A, B, C, D)

    # Metadata MLP
    META_FEATURES = ["age", "implant", "laterality", "view", "site_id"]
    META_EMBED_DIM = 32

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_EPOCHS = 10
    BATCH_SIZE = 12  # Adjusted for A100 40GB with 768x768 resolution
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0
    PATIENCE = 3  # Early stopping patience

    # Loss Weights
    POS_WEIGHT = 15.0  # Weight for positive class in BCE (Imbalance ~1:50)
    LAMBDA_BIRADS = 0.5  # Weight for BIRADS auxiliary loss
    LAMBDA_DENSITY = 0.5  # Weight for Density auxiliary loss

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Data Mappings (Categorical Encoding)
    # =========================================================================
    # These mappings ensure consistent encoding between Train/Val/Test

    # Laterality: Left vs Right
    LATERALITY_MAP = {"L": 0, "R": 1}

    # View: Orientation of the mammogram
    # Common views: CC (Craniocaudal), MLO (Mediolateral oblique)
    # Rare views: LM, LMO, AT, ML
    VIEW_MAP = {"CC": 0, "MLO": 1, "LM": 2, "LMO": 3, "AT": 4, "ML": 5}

    # Site ID: Hospital ID
    SITE_ID_MAP = {1: 0, 2: 1}

    # Density: Breast density (A=Least dense, D=Most dense)
    DENSITY_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

    # BIRADS: Assessment category
    # 0: Incomplete/Follow-up, 1: Negative, 2: Benign
    BIRADS_MAP = {0: 0, 1: 1, 2: 2}

    @classmethod
    def get_transforms(cls, data_split="train"):
        """
        Returns the appropriate Albumentations transforms for the split.
        Note: Requires albumentations to be imported where used.
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if data_split == "train":
            return A.Compose(
                [
                    A.Resize(height=cls.IMG_SIZE[0], width=cls.IMG_SIZE[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                            A.ElasticTransform(
                                alpha=1, sigma=50, alpha_affine=50, p=1.0
                            ),
                        ],
                        p=0.3,
                    ),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(height=cls.IMG_SIZE[0], width=cls.IMG_SIZE[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
