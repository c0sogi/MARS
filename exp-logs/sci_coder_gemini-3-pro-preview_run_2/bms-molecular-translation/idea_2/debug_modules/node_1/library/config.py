import os
import torch


class Config:
    """
    Centralized configuration for the Global Context Image-to-Sequence Network.
    """

    # -------------------------------------------------------------------------
    # General Compute Settings
    # -------------------------------------------------------------------------
    seed = 42
    debug = False  # Set True to restrict dataset size for debugging
    num_workers = 12  # Utilizing available vCPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    input_dir = "./input"
    metadata_dir = "./metadata"

    # Metadata paths (pre-generated)
    train_metadata_path = os.path.join(metadata_dir, "train.csv")
    val_metadata_path = os.path.join(metadata_dir, "val.csv")
    test_metadata_path = os.path.join(metadata_dir, "test.csv")

    # Working directory for caching and artifacts
    # Strictly following the requirement to use ./working/idea_2/
    working_dir = "./working/idea_2"
    os.makedirs(working_dir, exist_ok=True)

    # Model checkpoint path
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # Cache paths for deterministic data processing
    tokenizer_cache_path = os.path.join(working_dir, "tokenizer_vocab.npy")

    # Submission directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # -------------------------------------------------------------------------
    # Data & Preprocessing
    # -------------------------------------------------------------------------
    image_size = 224  # Fixed square resolution for MobileNetV2
    input_channels = 3  # RGB images

    # Max sequence length. EDA showed max InChI length is 403.
    # We add a small buffer for special tokens (<SOS>, <EOS>).
    max_length = 410

    # -------------------------------------------------------------------------
    # Tokenizer / Vocabulary
    # -------------------------------------------------------------------------
    sos_token = "<SOS>"
    eos_token = "<EOS>"
    pad_token = "<PAD>"
    unk_token = "<UNK>"

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Encoder: Lightweight CNN backbone
    encoder_name = "mobilenet_v2"

    # Decoder: LSTM
    embedding_dim = 256  # Dimension for character embeddings
    decoder_hidden_dim = 512  # LSTM hidden state size
    decoder_layers = 1  # Number of LSTM layers
    dropout = 0.5  # Dropout probability

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    epochs = 15  # Sufficient for convergence with pre-trained backbone
    batch_size = 128  # A100 GPU has 40GB VRAM, allows larger batches

    # Optimization
    encoder_lr = 1e-4  # Lower learning rate for fine-tuning CNN
    decoder_lr = 4e-4  # Higher learning rate for training LSTM from scratch
    weight_decay = 1e-6
    clip_grad_norm = 5.0  # Gradient clipping to prevent exploding gradients in LSTM

    # Learning Rate Scheduler
    scheduler_factor = 0.5
    scheduler_patience = 2
    min_lr = 1e-6

    # Early Stopping
    patience = 5  # Stop if validation metric doesn't improve

    # Training Strategy
    teacher_forcing_ratio = (
        0.5  # Probability of using ground truth as input during training
    )

    def __init__(self, debug=False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, overrides settings for quick debugging.
        """
        if debug:
            self.debug = True
            self.epochs = 2
            self.batch_size = 16
            self.num_workers = 0  # Avoid multiprocessing overhead in debug
            print("Config initialized in DEBUG mode.")
