import os
import random
import numpy as np
import torch
from pathlib import Path

# ==========================================
# 1. Global Configuration & Reproducibility
# ==========================================
SEED = 42


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enabled for speed on A100


# ==========================================
# 2. Directory Structure
# ==========================================
class Paths:
    # Read-only inputs
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")

    # Working directory for caching intermediate files (Idea 6)
    WORKING_DIR = Path("./working/idea_6")

    # Final submission output
    SUBMISSION_DIR = Path("./submission")

    @classmethod
    def setup(cls):
        """Creates necessary output directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize paths immediately
Paths.setup()

# ==========================================
# 3. Data & Time Window Configuration
# ==========================================
DATA_CONFIG = {
    # Validation Strategy: Last 7 days of training data
    "val_days": 7,
    # Source B: Linear-Decay Co-occurrence (Local Structure)
    # Restricted to recent 4 weeks to minimize noise
    "cooc_window_days": 28,
    # Source A: LightGCN Graph (High-Order Structure)
    # Uses 12 weeks to capture broader connectivity
    "graph_window_days": 84,
    # Prediction horizon
    "test_days": 7,
}

# ==========================================
# 4. Model Hyperparameters
# ==========================================

# Source A: LightGCN (Graph Collaborative Filtering)
GCN_PARAMS = {
    "embedding_dim": 64,
    "n_layers": 3,  # Standard LightGCN depth
    "batch_size": 4096,  # Optimized for A100 40GB
    "lr": 1e-3,
    "epochs": 20,
    "decay": 1e-4,  # L2 Regularization
    "neg_samples": 10,  # 1 Positive : 10 Negatives (Robustness)
    "top_k_retrieval": 20,  # Candidates to retrieve per user from Graph
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
}

# Stage 2: Interaction-Aware Ranker (LightGBM)
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": 12,
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,  # High capacity configuration
    "num_leaves": 128,  # High capacity configuration
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "early_stopping_rounds": 50,
    "seed": SEED,
    "n_jobs": 12,  # Use available vCPUs
}

# ==========================================
# 5. Candidate Generation Configuration
# ==========================================
CANDIDATE_CONFIG = {
    # Number of candidates to retrieve from each source
    "top_k_cooc": 12,  # From Item-Item Co-occurrence
    "top_k_graph": 20,  # From LightGCN (matches GCN_PARAMS)
    "top_k_repurchase": 12,  # From User History
    "top_k_popular": 12,  # Fallback (Recent Popularity)
}
