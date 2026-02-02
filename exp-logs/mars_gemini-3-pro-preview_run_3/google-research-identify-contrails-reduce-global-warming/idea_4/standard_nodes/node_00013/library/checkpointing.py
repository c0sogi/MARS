import os
import glob
import re
import torch
from library.config import Config


class CheckpointManager:
    """
    Manages saving and pruning of model checkpoints during training.
    Maintains the top K models based on a specified metric (Dice Score).
    """

    def __init__(self, checkpoint_dir=None, top_k=Config.N_BEST_MODELS, mode="max"):
        """
        Args:
            checkpoint_dir (str): Directory to save checkpoints. Defaults to Config working dir.
            top_k (int): Number of best checkpoints to keep.
            mode (str): 'max' for metrics like Dice (higher is better), 'min' for Loss.
        """
        self.checkpoint_dir = (
            checkpoint_dir
            if checkpoint_dir
            else os.path.join(Config.WORKING_DIR, "checkpoints")
        )
        self.top_k = top_k
        self.mode = mode
        self.checkpoints = []  # List of tuples: (score, epoch, filepath)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, epoch, score):
        """
        Saves the model if it qualifies as one of the top K models.
        Prunes the worst model if the limit is exceeded.

        Args:
            model (nn.Module): The model to save.
            epoch (int): Current epoch.
            score (float): Metric score (Dice).
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{score:.6f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Save the current model state
        # Handle DataParallel or DistributedDataParallel wrappers if present
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, filepath)

        # Add to the list
        self.checkpoints.append((score, epoch, filepath))

        # Sort the list
        # If mode is 'max' (Dice), sort descending (best first)
        # If mode is 'min' (Loss), sort ascending (best first)
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Prune if we have more than K models
        if len(self.checkpoints) > self.top_k:
            # The last element is the "worst" based on the sort order
            worst_score, worst_epoch, worst_path = self.checkpoints.pop()

            # Delete the file
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                except OSError as e:
                    print(f"Error removing checkpoint {worst_path}: {e}")

    def load_best(self):
        """
        Loads the state dict of the best model currently tracked.

        Returns:
            dict: State dictionary of the best model.
        """
        if not self.checkpoints:
            print("No checkpoints available to load.")
            return None

        # Best model is at index 0
        best_score, best_epoch, best_path = self.checkpoints[0]
        print(
            f"Loading best checkpoint from Epoch {best_epoch} (Score: {best_score:.6f})"
        )

        return torch.load(best_path, map_location=Config.DEVICE)


def average_weights(
    checkpoint_dir=None, threshold_epoch=Config.CONVERGENCE_EPOCH_THRESHOLD
):
    """
    Averages the weights of checkpoints saved strictly AFTER the threshold epoch.
    Only averages floating-point parameters to protect integer buffers (e.g., BN stats).

    Args:
        checkpoint_dir (str): Directory containing .pth files.
        threshold_epoch (int): Epoch number; only checkpoints with epoch > threshold are used.

    Returns:
        dict: The averaged state dictionary.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    # List all .pth files
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    # Regex to extract epoch number from filename
    # Expected format: checkpoint_epoch_{epoch}_dice_{score}.pth
    epoch_pattern = re.compile(r"checkpoint_epoch_(\d+)_")

    valid_checkpoints = []

    print(f"Scanning for checkpoints to average (Epoch > {threshold_epoch})...")

    for f in files:
        match = epoch_pattern.search(os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            if epoch > threshold_epoch:
                valid_checkpoints.append(f)

    if not valid_checkpoints:
        print("No checkpoints found meeting the convergence criteria.")
        return None

    print(
        f"Averaging weights from {len(valid_checkpoints)} checkpoints: {[os.path.basename(f) for f in valid_checkpoints]}"
    )

    # Load the first checkpoint to serve as the base structure
    # We use CPU to avoid GPU OOM during the averaging process
    avg_state_dict = torch.load(valid_checkpoints[0], map_location="cpu")

    # Identify floating point keys
    float_keys = [k for k, v in avg_state_dict.items() if torch.is_floating_point(v)]

    # Accumulate weights
    # We start with the values from the first checkpoint already loaded
    # So we loop through the rest and add
    if len(valid_checkpoints) > 1:
        for i in range(1, len(valid_checkpoints)):
            state_dict = torch.load(valid_checkpoints[i], map_location="cpu")
            for k in float_keys:
                avg_state_dict[k] += state_dict[k]

        # Divide by number of models
        n = len(valid_checkpoints)
        for k in float_keys:
            avg_state_dict[k] = avg_state_dict[k] / n

    print("Weight averaging completed successfully.")
    return avg_state_dict
