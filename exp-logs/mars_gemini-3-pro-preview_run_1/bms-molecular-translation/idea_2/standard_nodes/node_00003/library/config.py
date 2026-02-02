import os
import torch


class Config:
    """
    Configuration class for the Image-to-InChI prediction model.
    This class holds all hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False, epochs=6, batch_size=64, image_size=256):
        # General Settings
        self.seed = 42
        self.debug = debug
        self.num_workers = 4

        # Directory Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.work_dir = "./working/idea_2"

        # Ensure working directory exists
        os.makedirs(self.work_dir, exist_ok=True)

        # File Paths
        self.train_metadata_path = os.path.join(self.metadata_dir, "train_metadata.csv")
        self.val_metadata_path = os.path.join(self.metadata_dir, "val_metadata.csv")
        self.test_metadata_path = os.path.join(self.metadata_dir, "test_metadata.csv")
        self.sample_submission_path = os.path.join(
            self.input_dir, "sample_submission.csv"
        )
        self.tokenizer_path = os.path.join(self.work_dir, "tokenizer.npy")
        self.model_save_path = os.path.join(self.work_dir, "best_model.pth")
        self.checkpoint_path = os.path.join(self.work_dir, "checkpoint.pth")
        self.submission_path = "./submission/submission.csv"

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)

        # Data Parameters
        self.image_size = (image_size, image_size)
        self.max_len = 410  # Based on EDA max length of 403 + buffer

        # Model Architecture Hyperparameters
        self.encoder_name = "efficientnet_b0"
        self.encoder_dim = 1280  # EfficientNet-B0 output channels
        self.embed_dim = 512  # Character embedding dimension
        self.decoder_dim = 512  # LSTM hidden state dimension
        self.attention_dim = 256  # Bahdanau attention dimension
        self.dropout = 0.5

        # Training Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        self.clip_grad = 5.0
        self.patience = 3  # Early stopping patience
        self.teacher_forcing_ratio = 0.5  # Initial ratio, can be scheduled

        # Debug Mode Overrides
        if self.debug:
            self.epochs = 1
            self.batch_size = 16
            self.subset_size = 1000  # Number of samples to use in debug mode
        else:
            self.subset_size = None  # Use full dataset

        # Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_config(self):
        """Prints the current configuration."""
        print("=" * 30)
        print("Current Configuration:")
        print(f"  Debug Mode:       {self.debug}")
        print(f"  Device:           {self.device}")
        print(f"  Image Size:       {self.image_size}")
        print(f"  Batch Size:       {self.batch_size}")
        print(f"  Epochs:           {self.epochs}")
        print(f"  Max Seq Len:      {self.max_len}")
        print(f"  Encoder:          {self.encoder_name}")
        print(f"  Decoder Dim:      {self.decoder_dim}")
        print(f"  Attention Dim:    {self.attention_dim}")
        print(f"  Work Dir:         {self.work_dir}")
        print("=" * 30)
