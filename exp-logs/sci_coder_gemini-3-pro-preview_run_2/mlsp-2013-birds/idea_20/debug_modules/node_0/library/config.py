import os
import torch
import numpy as np


class Config:
    # --------------------------------------------------------------------------
    # Experiment Control
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    # Image Dimensions: (Height, Width) -> (Frequency, Time)
    # We use a 1:2 aspect ratio as found effective in previous experiments.
    IMG_SIZE = (224, 448)

    # Input Channels: 3 (Pseudo-RGB for ImageNet pretrained models)
    CHANNELS = 3

    # Number of target classes
    NUM_SPECIES = 19

    # Source Data: Use Filtered Spectrograms (denoised)
    USE_FILTERED_SPECTROGRAMS = True

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Heterogeneous Ensemble to maximize diversity and stability
    MODELS = ["resnet18", "efficientnet_b0", "resnet34"]

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 25
    BATCH_SIZE = 16

    # Optimizer Settings (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # --------------------------------------------------------------------------
    # Regularization & Augmentation
    # --------------------------------------------------------------------------
    # Mixup Augmentation
    MIXUP_ALPHA = 0.4

    # Siamese Temporal Consistency Regularization
    # Weight for the MSE loss between predictions of original and time-rolled inputs
    CONSISTENCY_LAMBDA = 2.0

    # Hardware Settings
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Logic
    # --------------------------------------------------------------------------
    @staticmethod
    def get_pos_weights(df):
        """
        Calculates positive weights for BCEWithLogitsLoss to handle class imbalance.
        Formula: pos_weight = number_of_negatives / number_of_positives

        Args:
            df (pd.DataFrame): DataFrame containing the training data.
                               Must have columns starting with 'species_'.

        Returns:
            torch.Tensor: A tensor of weights for each class.
        """
        # Identify label columns
        label_cols = [c for c in df.columns if c.startswith("species_")]

        # Extract labels
        labels = df[label_cols].values

        # Calculate counts for positives and negatives
        pos_counts = np.sum(labels, axis=0)
        total_samples = len(labels)
        neg_counts = total_samples - pos_counts

        # Clip positive counts to 1 to prevent division by zero
        pos_counts = np.maximum(pos_counts, 1)

        # Compute weights
        weights = neg_counts / pos_counts

        return torch.tensor(weights, dtype=torch.float32)
