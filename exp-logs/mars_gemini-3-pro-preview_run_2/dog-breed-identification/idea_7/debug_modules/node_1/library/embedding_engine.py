import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from library.config import Config


def set_seed(seed=Config.SEED):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    """
    Creates the transformation pipelines for the three views.
    Returns a dictionary of transform functions/objects.
    """
    # Standard ImageNet normalization
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    # View 1: Global (Squish)
    global_trans = T.Compose([T.Resize(Config.GLOBAL_VIEW_SIZE), to_tensor, normalize])

    # View 2: Standard (Context)
    standard_trans = T.Compose(
        [
            T.Resize(Config.STANDARD_RESIZE),
            T.CenterCrop(Config.STANDARD_CROP),
            to_tensor,
            normalize,
        ]
    )

    # View 3: Robust Local (Texture) components
    # We apply these manually to handle the FiveCrop tuple output
    local_resize = T.Resize(Config.LOCAL_RESIZE)
    five_crop = T.FiveCrop(Config.LOCAL_CROP)

    return {
        "global": global_trans,
        "standard": standard_trans,
        "local_resize": local_resize,
        "five_crop": five_crop,
        "norm": normalize,
        "to_tensor": to_tensor,
    }


def prepare_batch_views(images, transforms):
    """
    Processes a list of PIL images into a dictionary of tensors for all views and TTA.

    Args:
        images (list): List of PIL Images.
        transforms (dict): Dictionary of transform objects.

    Returns:
        dict: Dictionary containing stacked tensors for original and flipped versions of all views.
    """
    global_imgs = []
    global_imgs_flip = []
    std_imgs = []
    std_imgs_flip = []
    local_imgs = []  # Will store stacked crops (5, C, H, W)
    local_imgs_flip = []  # Will store stacked crops (5, C, H, W)

    g_trans = transforms["global"]
    s_trans = transforms["standard"]
    l_resize = transforms["local_resize"]
    five_crop = transforms["five_crop"]
    norm = transforms["norm"]
    to_tensor = transforms["to_tensor"]

    for img in images:
        # --- Global View ---
        global_imgs.append(g_trans(img))
        global_imgs_flip.append(g_trans(TF.hflip(img)))

        # --- Standard View ---
        std_imgs.append(s_trans(img))
        std_imgs_flip.append(s_trans(TF.hflip(img)))

        # --- Local View ---
        # 1. Resize
        img_r = l_resize(img)
        img_r_f = l_resize(TF.hflip(img))

        # 2. FiveCrop (returns tuple of 5 PIL images)
        crops = five_crop(img_r)
        crops_f = five_crop(img_r_f)

        # 3. Process Crops: ToTensor -> Normalize -> Stack
        c_t = torch.stack([norm(to_tensor(c)) for c in crops])  # (5, 3, 224, 224)
        c_t_f = torch.stack([norm(to_tensor(c)) for c in crops_f])  # (5, 3, 224, 224)

        local_imgs.append(c_t)
        local_imgs_flip.append(c_t_f)

    # Stack lists into batch tensors
    batch_data = {
        "global": torch.stack(global_imgs),  # (B, 3, 224, 224)
        "global_flip": torch.stack(global_imgs_flip),
        "std": torch.stack(std_imgs),  # (B, 3, 224, 224)
        "std_flip": torch.stack(std_imgs_flip),
        "local": torch.stack(local_imgs),  # (B, 5, 3, 224, 224)
        "local_flip": torch.stack(local_imgs_flip),
    }
    return batch_data


def extract_embeddings(
    dataloader,
    model,
    device,
    cache_path_features,
    cache_path_targets,
    load_cached_data=True,
):
    """
    Main function to run the Multi-Scale Deep Feature Pyramid pipeline.

    Args:
        dataloader: PyTorch DataLoader yielding (images, targets).
        model: The ConvNeXtDualExtractor model.
        device: 'cuda' or 'cpu'.
        cache_path_features: Path to save/load features .npy.
        cache_path_targets: Path to save/load targets .npy.
        load_cached_data: Boolean flag to enable/disable loading from cache.

    Returns:
        tuple: (features_numpy, targets_numpy)
    """
    set_seed()

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_features)
        and os.path.exists(cache_path_targets)
    ):
        print(f"Loading cached embeddings from {cache_path_features}...")
        features = np.load(cache_path_features)
        targets = np.load(
            cache_path_targets, allow_pickle=True
        )  # allow_pickle for string IDs
        return features, targets

    print("Starting Multi-Scale Multi-View Feature Extraction...")

    # 2. Setup
    model.to(device)
    model.eval()
    transforms = get_transforms()

    all_features = []
    all_targets = []

    # 3. Processing Loop
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # images is a list of PIL images
            # targets is Tensor(int) for train/val, or tuple(str) for test

            # A. Prepare Views (CPU intensive, done per batch)
            inputs = prepare_batch_views(images, transforms)

            # B. Move to Device
            # Global & Standard: (B, 3, H, W)
            # Local: (B, 5, 3, H, W)
            g_img = inputs["global"].to(device)
            g_img_f = inputs["global_flip"].to(device)
            s_img = inputs["std"].to(device)
            s_img_f = inputs["std_flip"].to(device)
            l_img = inputs["local"].to(device)
            l_img_f = inputs["local_flip"].to(device)

            batch_size = g_img.size(0)

            # C. Inference Helper
            def get_feats(x):
                # Returns dict {'stage3': ..., 'stage4': ...}
                return model(x)

            # --- Global View Inference ---
            out_g = get_feats(g_img)
            out_g_f = get_feats(g_img_f)

            # TTA Average
            g_s4 = (out_g["stage4"] + out_g_f["stage4"]) / 2.0
            g_s3 = (out_g["stage3"] + out_g_f["stage3"]) / 2.0

            # --- Standard View Inference ---
            out_s = get_feats(s_img)
            out_s_f = get_feats(s_img_f)

            # TTA Average
            s_s4 = (out_s["stage4"] + out_s_f["stage4"]) / 2.0
            s_s3 = (out_s["stage3"] + out_s_f["stage3"]) / 2.0

            # --- Local View Inference ---
            # Input is (B, 5, 3, H, W). Flatten to (B*5, 3, H, W) for batch processing
            l_flat = l_img.view(-1, 3, 224, 224)
            l_flat_f = l_img_f.view(-1, 3, 224, 224)

            out_l = get_feats(l_flat)
            out_l_f = get_feats(l_flat_f)

            # TTA Average (still flattened)
            l_s4_flat = (out_l["stage4"] + out_l_f["stage4"]) / 2.0
            l_s3_flat = (out_l["stage3"] + out_l_f["stage3"]) / 2.0

            # Reshape back to (B, 5, D) and Average over crops
            # stage4
            dim_s4 = l_s4_flat.size(1)
            l_s4 = l_s4_flat.view(batch_size, 5, dim_s4).mean(dim=1)

            # stage3
            dim_s3 = l_s3_flat.size(1)
            l_s3 = l_s3_flat.view(batch_size, 5, dim_s3).mean(dim=1)

            # D. Concatenation (Early Fusion)
            # Order: Global_S4, Global_S3, Standard_S4, Standard_S3, Local_S4, Local_S3
            final_vec = torch.cat([g_s4, g_s3, s_s4, s_s3, l_s4, l_s3], dim=1)

            # E. Store
            all_features.append(final_vec.cpu().numpy())

            # Handle targets (convert tensor to numpy or tuple to numpy)
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.numpy())
            else:
                all_targets.append(np.array(list(targets)))

    # 4. Aggregate Results
    features_arr = np.concatenate(all_features, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    print(f"Extraction Complete. Feature Shape: {features_arr.shape}")

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path_features), exist_ok=True)

    np.save(cache_path_features, features_arr)
    np.save(cache_path_targets, targets_arr)
    print(f"Saved features to {cache_path_features}")

    return features_arr, targets_arr
