import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.
    Ensures inputs are flattened numpy arrays.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return roc_auc_score(y_true, y_pred)


def init_weights(m):
    """
    Primary weight initialization function for the Interface-Normalized Hybrid SwiGLU Network.

    Strategies:
    - Linear (Backbone): Kaiming (He) Uniform.
    - Embedding (Tokens): Unit Variance (std=1.0).
    - Embedding (Positions): Low Variance (std=0.02).
    - LayerNorm: Weight=1, Bias=0.
    """
    if isinstance(m, nn.Linear):
        # SwiGLU Backbone uses Kaiming Uniform
        # Using nonlinearity='relu' as a proxy for Swish/GELU-like activations in He init
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Embedding):
        # Distinguish based on dimensions defined in Config
        if m.num_embeddings == Config.CAT_VOCAB_SIZE:
            # Token Embeddings: Unit Variance
            nn.init.normal_(m.weight, mean=0.0, std=Config.EMBED_INIT_STD)
        elif m.num_embeddings == Config.CAT_SEQ_LEN:
            # Positional Embeddings (if nn.Embedding): Low Variance
            nn.init.normal_(m.weight, mean=0.0, std=Config.POS_EMBED_INIT_STD)
        else:
            # Default fallback for other embeddings
            nn.init.normal_(m.weight, mean=0.0, std=1.0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def init_transformer_weights(m):
    """
    Specialized initialization for Transformer components.

    Strategies:
    - Linear: Xavier (Glorot) Uniform.
    - LayerNorm: Weight=1, Bias=0.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def init_pos_embed(tensor):
    """
    Helper to initialize positional embeddings if they are stored as raw Parameters.
    Strategy: Low Variance Random Noise (std=0.02).
    """
    nn.init.normal_(tensor, mean=0.0, std=Config.POS_EMBED_INIT_STD)
