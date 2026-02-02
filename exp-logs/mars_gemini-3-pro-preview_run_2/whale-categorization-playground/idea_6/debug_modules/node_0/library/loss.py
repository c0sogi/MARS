import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) for deep face recognition.

    This loss function improves the discriminative power of the learned embeddings
    by maximizing the inter-class variance and minimizing the intra-class variance
    on a hypersphere manifold.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition". CVPR 2019.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features (int): Size of each input sample (embedding dimension).
            out_features (int): Size of each output sample (number of classes).
            s (float): Norm of input feature (scale). Default: 30.0.
            m (float): Margin value. Default: 0.50.
            easy_margin (bool): If True, use easy margin to prevent training instability.
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Weight parameter: (out_features, in_features)
        # We use a Parameter so it is registered in the module and optimized.
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Thresholds for stability check
        # th: cos(pi - m)
        self.th = math.cos(math.pi - m)
        # mm: sin(pi - m) * m
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """
        Args:
            input (torch.Tensor): Feature embeddings. Shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels. Shape (batch_size,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # ---------------------------
        # 1. Normalize Inputs & Weights
        # ---------------------------
        # Normalize features (x)
        # input shape: (B, D)
        x = F.normalize(input)

        # Normalize weights (W)
        # weight shape: (C, D)
        W = F.normalize(self.weight)

        # ---------------------------
        # 2. Compute Cosine Similarity
        # ---------------------------
        # cosine = x . W^T
        # Shape: (B, C)
        cosine = F.linear(x, W)

        # ---------------------------
        # 3. Additive Angular Margin
        # ---------------------------
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)

        # Calculate sin(theta) = sqrt(1 - cos(theta)^2)
        # Clamp cosine to ensure numerical stability for sqrt (avoid negative due to precision)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # If easy_margin is True, we don't enforce the margin when theta is large
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # When theta > pi - m, cos(theta + m) is not monotonic w.r.t theta.
            # We use a fallback (cosine - mm) in that region to maintain a penalty gradient.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # ---------------------------
        # 4. Apply Margin to Target Class
        # ---------------------------
        # We only want to modify the logit corresponding to the ground truth label.

        # Convert labels to long for indexing
        label = label.long()

        # Create output tensor (logits) starting as a copy of cosine
        output = cosine * 1.0

        # Get batch indices [0, 1, ..., B-1]
        batch_size = len(input)
        batch_indices = torch.arange(batch_size, device=input.device)

        # Replace target logits with margin-adjusted logits (phi)
        # output[i, label[i]] = phi[i, label[i]]
        output[batch_indices, label] = phi[batch_indices, label]

        # ---------------------------
        # 5. Scale and Cross Entropy
        # ---------------------------
        output *= self.s

        loss = F.cross_entropy(output, label)

        return loss
