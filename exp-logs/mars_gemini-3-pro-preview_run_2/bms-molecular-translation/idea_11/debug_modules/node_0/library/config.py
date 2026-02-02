import os
import torch


class Config:
    """
    Configuration class for the Anisotropic ResNet-Transformer with 2D Spatial-Awareness.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "inchi_prediction_idea_11"
    DEBUG = False
    SEED = 42
    NUM_WORKERS = 12  # Utilizing all available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"

    # Working directory for checkpoints, logs, and cached data
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Path to save/load the tokenizer vocabulary
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    # Fixed height resizing strategy to maintain vertical atom scale.
    # 320px is chosen to ensure atoms are distinguishable after 32x downsampling.
    IMAGE_HEIGHT = 320

    # Max width to clip extremely wide images to prevent OOM errors.
    # Based on EDA, max width is ~2581. Resizing H 220->320 scales W by ~1.45.
    # 2581 * 1.45 ~= 3742. We set a safe upper bound.
    MAX_WIDTH = 3840

    # Pad width to be a multiple of this value (Encoder stride).
    # Standard ResNet stride is 32.
    PAD_MULTIPLE = 32

    # InChI string parameters.
    # Max length observed in EDA is 403. We add a buffer for <SOS>, <EOS>.
    MAX_TEXT_LEN = 450

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Encoder
    ENCODER_NAME = "resnet50"
    ENCODER_DIM = 2048  # Output channels of ResNet50 bottleneck

    # Decoder (Transformer)
    DECODER_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 4
    FF_DIM = 2048
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Adjusted for A100 40GB with larger image sizes
    EPOCHS = 15

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 5.0

    # Scheduler (OneCycleLR)
    PCT_START = 0.1
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1000

    # Early Stopping
    PATIENCE = 5

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    BEAM_SIZE = 3  # Beam width for beam search decoding

    @classmethod
    def setup(cls, debug: bool = False, epochs: int = None, batch_size: int = None):
        """
        Configure the experiment settings.

        Args:
            debug (bool): If True, enables debug mode (fewer epochs, smaller batches, subset data).
            epochs (int, optional): Override the number of epochs.
            batch_size (int, optional): Override the batch size.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 16
            # Use a separate directory for debug artifacts to avoid overwriting production runs
            cls.WORKING_DIR = "./working/idea_11_debug"
            os.makedirs(cls.WORKING_DIR, exist_ok=True)
            print(
                f"[Config] Debug mode enabled. Epochs={cls.EPOCHS}, BatchSize={cls.BATCH_SIZE}"
            )

        if epochs is not None:
            cls.EPOCHS = epochs

        if batch_size is not None:
            cls.BATCH_SIZE = batch_size
