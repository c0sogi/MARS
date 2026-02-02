import os
import random
import numpy as np
import torch
import shutil


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(
    state,
    is_best,
    checkpoint_dir="./working/idea_25/checkpoints",
    filename="checkpoint.pth",
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str or torch.device): Device to map the location to.

    Returns:
        epoch (int): The epoch at which the checkpoint was saved.
        best_metric (float): The best metric value recorded.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    # Handle DataParallel wrapping if necessary
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)

    return epoch, best_metric


def moving_average(net1, net2, alpha=1.0):
    """
    Computes the moving average of the parameters of two networks.
    net1 = (1-alpha)*net1 + alpha*net2

    Used for SWA.
    """
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def update_bn(loader, model, device="cuda"):
    """
    Updates the BatchNorm running statistics by making a forward pass
    on the training data. Essential for SWA models.

    Args:
        loader (torch.utils.data.DataLoader): Training data loader.
        model (torch.nn.Module): The model to update.
        device (str): Device to run the forward pass on.
    """
    model.train()
    momenta = {}

    # Temporarily disable momentum for BN updates
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    with torch.no_grad():
        # Cite debug_lesson_7: Handle Tuple Nesting Correctly When Unpacking enumerate
        # Loader yields (img, target, rec_id), so we unpack carefully or index into batch
        for i, batch in enumerate(loader):
            input_data = batch[0].to(device)
            model(input_data)

    # Restore momentum
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]
