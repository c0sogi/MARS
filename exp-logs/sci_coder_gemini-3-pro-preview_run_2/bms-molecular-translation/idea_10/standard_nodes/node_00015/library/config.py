import os
import torch


class Config:
    """
    Configuration class for Idea 10: Anisotropic Hybrid ResNet-Transformer.
    """

    def __init__(self, debug: bool = False):
        # ---------------------------------------------------------
        # General Settings
        # ---------------------------------------------------------
        self.seed = 42
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4 if not debug else 0

        # ---------------------------------------------------------
        # Data Parameters
        # ---------------------------------------------------------
        # Images are resized to a fixed height while maintaining aspect ratio.
        # Grayscale conversion is applied (1 channel).
        self.image_height = 192
        self.input_channels = 1

        # The model is designed to have a horizontal stride of 4.
        # We pad images to be multiples of this stride to ensure valid feature map dimensions.
        self.horizontal_stride = 4
        self.vertical_stride = 32  # ResNet default total stride for height

        # Max sequence length for the Transformer (InChI strings)
        # 99th percentile is ~223, max is ~403.
        # We add some buffer for special tokens and potential long tails.
        self.max_len = 512

        # ---------------------------------------------------------
        # Paths
        # ---------------------------------------------------------
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_10"
        self.submission_dir = "./submission"

        self.train_metadata_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test.csv")

        self.vocab_path = os.path.join(self.working_dir, "vocab.npy")
        self.model_path = os.path.join(self.working_dir, "best_model.pth")
        self.checkpoint_path = os.path.join(self.working_dir, "checkpoint.pth")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Ensure working directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Model Architecture
        # ---------------------------------------------------------
        self.resnet_arch = "resnet34"

        # Transformer / Hybrid parameters
        self.encoder_dim = (
            384  # Dimension of the feature sequence after ResNet projection
        )
        self.decoder_dim = 384  # Dimension of the Transformer Decoder
        self.nhead = 8  # Number of attention heads
        self.num_encoder_layers = (
            2  # Layers in the Sequence Encoder (e.g., Transformer Encoder or GRU)
        )
        self.num_decoder_layers = 4  # Layers in the Transformer Decoder
        self.dim_feedforward = 1024
        self.dropout = 0.1

        # ---------------------------------------------------------
        # Training Hyperparameters
        # ---------------------------------------------------------
        self.epochs = 5 if not debug else 2
        self.batch_size = 32 if not debug else 8

        # Optimization
        self.learning_rate = 3e-4
        self.weight_decay = 1e-2
        self.max_grad_norm = 10.0

        # Loss Weights
        # Joint Loss = lambda * CTC + (1 - lambda) * CE
        self.ctc_weight = 0.5

        # Scheduler
        self.pct_start = 0.1  # Percentage of training to increase LR (OneCycle)

        # Early Stopping
        self.patience = 3

        # ---------------------------------------------------------
        # Inference
        # ---------------------------------------------------------
        self.beam_size = 5
        self.print_freq = 100  # Print training logs every N steps

    def __repr__(self):
        return str(self.__dict__)
