import os
import torch


class PathConfig:
    """
    Defines file paths and directory structures for the project.
    """

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Input Files (Generated Metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Artifact Directories (Working)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MLM_MODELS_DIR = os.path.join(WORKING_DIR, "mlm_models")
    FINETUNED_MODELS_DIR = os.path.join(WORKING_DIR, "finetuned_models")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

    @classmethod
    def create_dirs(cls):
        """Creates necessary working directories if they don't exist."""
        directories = [
            cls.WORKING_DIR,
            cls.SUBMISSION_DIR,
            cls.CACHE_DIR,
            cls.MLM_MODELS_DIR,
            cls.FINETUNED_MODELS_DIR,
            cls.PREDICTIONS_DIR,
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)


class ModelConfig:
    """
    Defines model architecture and configuration.
    """

    # Selected Transformer Backbones
    BACKBONES = ["microsoft/deberta-v3-base", "roberta-base"]

    # Tokenizer & Input Settings
    MAX_LENGTH = 256  # Covers the vast majority of sentence lengths

    # Neural Architecture Settings
    USE_LAST_4_LAYERS = True  # Concatenate [CLS] from last 4 layers
    HIDDEN_DROPOUT = 0.1
    ATTENTION_DROPOUT = 0.1

    # Multi-Task Learning (MTL) Settings
    USE_MTL = True
    MTL_HEAD_DIM = 2  # Targets: [Log-Char-Length, Punctuation-Density]

    # Classification Labels
    LABELS = ["EAP", "HPL", "MWS"]
    NUM_LABELS = 3
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for i, l in enumerate(LABELS)}


class TrainConfig:
    """
    Defines training hyperparameters and strategies.
    """

    # General
    SEED = 42
    DEBUG = False  # Set to True to restrict dataset size for debugging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Domain-Adaptive Pre-training (Masked Language Modeling)
    DAPT_EPOCHS = 5
    DAPT_BATCH_SIZE = 16
    DAPT_LR = 2e-5
    DAPT_MASK_PROB = 0.15

    # Supervised Fine-Tuning (SFT)
    N_FOLDS = 5
    EPOCHS = 5
    BATCH_SIZE = 16
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 2  # Early stopping patience

    # Adversarial Weight Perturbation (AWP)
    USE_AWP = True
    AWP_START_EPOCH = 2
    AWP_LR = 1e-4
    AWP_EPS = 1e-2

    # Loss Optimization
    LAMBDA_MTL = 0.1  # Weight for the auxiliary regression loss

    # Learning Rate Scheduler
    SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1


class FeatureConfig:
    """
    Defines settings for statistical features and ensembling.
    """

    # TF-IDF Vectorization
    WORD_NGRAM_RANGE = (1, 2)
    CHAR_NGRAM_RANGE = (3, 5)
    MAX_FEATURES_WORD = 20000
    MAX_FEATURES_CHAR = 30000

    # Length-Adaptive Ensembling
    N_LENGTH_BINS = (
        3  # Split validation set into Short, Medium, Long for weight optimization
    )
