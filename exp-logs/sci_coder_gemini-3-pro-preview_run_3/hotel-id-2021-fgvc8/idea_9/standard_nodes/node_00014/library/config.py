import os
import torch


class Config:
    def __init__(self, debug: bool = False):
        """
        Configuration for the Hotel ID recognition task using EfficientNet-V2-M
        with a Multi-Stage Progressive Resolution pipeline.

        Args:
            debug (bool): If True, activates debug mode with reduced epochs and data.
        """
        # -------------------------------------------------------------------
        # General Settings
        # -------------------------------------------------------------------
        self.seed = 42
        self.num_workers = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        # -------------------------------------------------------------------
        # Directory Paths
        # -------------------------------------------------------------------
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.working_dir = "./working/idea_9"

        # Ensure working directory exists for caching/checkpoints
        os.makedirs(self.working_dir, exist_ok=True)

        # Metadata File Paths
        self.train_csv_path = os.path.join(self.metadata_dir, "train.csv")
        self.val_csv_path = os.path.join(self.metadata_dir, "val.csv")
        self.test_csv_path = os.path.join(self.metadata_dir, "test.csv")

        # Image Root Directories
        # Note: 'file_path' in metadata is relative to input_dir (e.g., train_images/0/xyz.jpg)
        self.image_root_dir = self.input_dir

        # Submission Output
        self.submission_dir = "./submission"
        os.makedirs(self.submission_dir, exist_ok=True)
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # -------------------------------------------------------------------
        # Model Architecture
        # -------------------------------------------------------------------
        # Backbone: EfficientNet-V2-M (Medium) provides better capacity than V2-S
        # while remaining faster than Large/XL variants.
        self.backbone_name = "tf_efficientnetv2_m"
        self.pretrained = True

        # Neck & Head Configuration
        self.embedding_dim = 512
        self.use_gem_pooling = True  # Generalized Mean Pooling
        self.use_bn_neck = True  # Batch Norm Neck

        # Classification Head: Sub-center ArcFace
        # Handles high intra-class variance better than standard ArcFace
        self.n_classes = 7770
        self.arcface_scale = 30.0
        self.arcface_margin = 0.50
        self.sub_centers_k = 3

        # -------------------------------------------------------------------
        # Training Hyperparameters (Multi-Stage Pipeline)
        # -------------------------------------------------------------------
        self.weight_decay = 1e-2

        # Stage 1: Low Resolution (224x224)
        # Objective: Fast convergence of global features and ArcFace margins.
        self.stage1_resolution = (224, 224)
        self.stage1_batch_size = 48  # Tuned for A100 40GB
        self.stage1_lr = 1e-3
        self.stage1_epochs = 8 if not self.debug else 1

        # Stage 2: High Resolution (384x384)
        # Objective: Resolve fine-grained details (text, logos).
        # Strategy: Reduced LR and Warmup to prevent gradient instability.
        self.stage2_resolution = (384, 384)
        self.stage2_batch_size = 24  # Reduced due to larger image size
        self.stage2_lr = 1e-4  # 10x lower than Stage 1
        self.stage2_epochs = 4 if not self.debug else 1
        self.stage2_warmup_epochs = 1

        # -------------------------------------------------------------------
        # Inference Hyperparameters
        # -------------------------------------------------------------------
        self.inference_resolution = (384, 384)
        self.inference_batch_size = 32
        self.use_tta = True  # Test-Time Augmentation (Horizontal Flip)
        self.top_k = 5  # Number of predictions per image
