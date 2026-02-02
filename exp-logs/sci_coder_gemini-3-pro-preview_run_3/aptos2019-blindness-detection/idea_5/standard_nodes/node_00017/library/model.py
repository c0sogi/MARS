import torch
import torch.nn as nn
import timm


def get_model(model_name="convnext_base", pretrained=True, checkpoint_path=None):
    """
    Creates a ConvNeXt model configured for regression (single scalar output).

    This architecture uses Layer Normalization instead of Batch Normalization,
    making it robust to small batch sizes required by high-resolution (768x768) inputs.

    Args:
        model_name (str): Name of the model architecture in timm (default: 'convnext_base').
        pretrained (bool): Whether to load ImageNet pretrained weights.
        checkpoint_path (str, optional): Path to a local checkpoint to load weights from.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    # Create the model using timm
    # num_classes=1 configures the final linear layer to output a single scalar
    # Use GeM pooling to better capture fine-grained lesions (Cite solution_lesson_node_00007)
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=1, global_pool="gem"
    )

    if checkpoint_path:
        # Load weights from the specified checkpoint
        # map_location='cpu' ensures we can load onto any machine,
        # the training loop will move it to GPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats (state_dict vs full checkpoint)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Clean state_dict keys to handle potential prefixes (e.g., 'module.' from DataParallel)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        # Load the state dictionary into the model
        # strict=False is used as a fallback if there are minor mismatches
        # (though for this specific task, strict=True is expected to work)
        try:
            model.load_state_dict(clean_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Retrying with strict=False...")
            model.load_state_dict(clean_state_dict, strict=False)

    return model
