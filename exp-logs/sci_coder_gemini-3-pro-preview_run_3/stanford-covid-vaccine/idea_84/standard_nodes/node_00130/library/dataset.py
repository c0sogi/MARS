import torch
from library.config import Config
from library.data_utils import RNADataset, get_dataloaders, load_data

# The RNADataset class is already fully implemented in library.data_utils
# and supports the required features (features, pair_indices, targets).
# We expose it here for the training script to import from 'dataset'.
RNADataset = RNADataset

# The get_dataloaders function handles:
# 1. Loading data (with caching logic implemented in load_data)
# 2. Creating RNADataset instances
# 3. Creating PyTorch DataLoaders with the correct batch size and workers
get_dataloaders = get_dataloaders

# We also expose load_data in case manual data inspection is needed
load_data = load_data
