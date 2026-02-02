import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from library.config import Config


class MLP(nn.Module):
    """Very simple Multi-Layer Perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DINOTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        # src: (S, B, C)
        q = k = src + pos if pos is not None else src
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DINOTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape[0])

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class DINOTransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        # tgt: (T, B, C), memory: (S, B, C)

        # Self Attention
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention
        q = tgt + query_pos if query_pos is not None else tgt
        k = memory + pos if pos is not None else memory
        tgt2 = self.multihead_attn(
            q,
            k,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DINOTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.norm1.normalized_shape[0])

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            intermediate.append(self.norm(output))

        # Return stack of intermediate outputs for auxiliary losses
        return torch.stack(intermediate)


class ContrastiveDeNoising(nn.Module):
    """
    Implements Contrastive DeNoising (CDN) for DINO.
    Generates noisy queries from ground truth to stabilize training.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_queries,
        num_patterns=3,
        label_noise_scale=0.2,
        box_noise_scale=0.4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_patterns = num_patterns  # Number of DN groups
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        # Embeddings for DN
        self.label_enc = nn.Embedding(
            num_classes + 1, hidden_dim
        )  # +1 for 'no object' if needed

    def forward(self, targets: List[Dict], label_enc_weight=None):
        """
        Args:
            targets: List of dicts containing 'labels' and 'boxes' (cx, cy, w, h)
            label_enc_weight: Shared weights from the class embedder of the main model
        Returns:
            input_query_content: (L, B, C)
            input_query_pos: (L, B, C)
            attn_mask: (L, L)
            dn_meta: Dict containing indices for loss computation
        """
        if not self.training or targets is None:
            return None, None, None, None

        device = targets[0]["boxes"].device
        batch_size = len(targets)

        # 1. Collect GT
        # Flatten batch
        known_labels = []
        known_boxes = []
        batch_idx = []

        for i, t in enumerate(targets):
            labels = t["labels"]
            boxes = t["boxes"]
            if len(labels) > 0:
                known_labels.append(labels)
                known_boxes.append(boxes)
                batch_idx.append(torch.full_like(labels, i))

        if len(known_labels) == 0:
            return None, None, None, None

        known_labels = torch.cat(known_labels)
        known_boxes = torch.cat(known_boxes)
        batch_idx = torch.cat(batch_idx)
        num_gt = known_labels.shape[0]

        # 2. Prepare DN queries (Repeat for num_patterns * 2 groups: positive and negative)
        # We create 2 groups per pattern: one positive (small noise), one negative (large noise)
        # Total DN queries = num_gt * num_patterns * 2

        # Repeat GT
        known_labels_ex = known_labels.repeat(self.num_patterns * 2, 1).view(-1)
        known_boxes_ex = known_boxes.repeat(self.num_patterns * 2, 1, 1).view(-1, 4)
        batch_idx_ex = batch_idx.repeat(self.num_patterns * 2, 1).view(-1)

        # 3. Apply Noise
        # Noise for boxes
        diff = torch.zeros_like(known_boxes_ex)
        diff[:, :2] = known_boxes_ex[:, 2:] * 0.5  # w/2, h/2
        diff[:, 2:] = known_boxes_ex[:, 2:] * 0.5  # w/2, h/2

        rand_sign = (
            torch.randint_like(known_boxes_ex, low=0, high=2, dtype=torch.float32) * 2.0
            - 1.0
        )
        rand_part = torch.rand_like(known_boxes_ex)
        rand_part[rand_part < 1e-5] = 1e-5  # avoid 0

        # Positive noise (small) vs Negative noise (large)
        # First half of patterns are positive, second half negative?
        # Actually DINO mixes them. Let's do simple: even indices positive, odd negative.
        # But simpler: First num_patterns groups are positive, next num_patterns are negative.

        num_dn_groups = self.num_patterns * 2

        # Create noise vector
        # Scale: Positive < 1.0 (e.g. 0.4), Negative > 1.0 (e.g. 2.0)?
        # Standard DINO: Positive < lambda, Negative > lambda.

        # We'll use a simplified heuristic:
        # Positive: noise < box_noise_scale
        # Negative: noise > box_noise_scale (or just random large noise)

        noise = rand_part * rand_sign

        # Create mask for negative samples (second half of the expansion)
        # Total size: num_patterns * 2 * num_gt
        # We want half to be positive, half negative

        # Indices
        total_dn_queries = known_boxes_ex.shape[0]
        # Split into 2 halves
        half = total_dn_queries // 2

        # Positive noise scaling
        noise[:half] *= self.box_noise_scale
        # Negative noise scaling (larger)
        noise[half:] *= self.box_noise_scale * 2.0

        known_boxes_noisy = known_boxes_ex + noise * diff
        known_boxes_noisy = known_boxes_noisy.clamp(min=0.0, max=1.0)

        # Noise for labels (flip labels for negative, keep for positive)
        # For negative queries, we might want to flip the label to something else or keep it.
        # DINO "CDN" uses "Contrastive":
        # Positive queries -> reconstruct GT.
        # Negative queries -> predict "No Object".

        known_labels_noisy = known_labels_ex.clone()
        # Randomly flip labels for negative group?
        # Actually, CDN relies on the box noise to make it negative. The label input is usually correct.
        # But we can add label noise too.
        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_noisy.float())
            # Flip labels in the negative part with higher probability?
            # For simplicity, we just apply random label flipping
            mask_label_noise = p < self.label_noise_scale
            # Assign random label
            new_labels = torch.randint_like(known_labels_noisy, 0, self.num_classes)
            known_labels_noisy[mask_label_noise] = new_labels[mask_label_noise]

        # 4. Embeddings
        # Content query: Class embedding
        if label_enc_weight is not None:
            # Use shared weights
            input_query_content = F.embedding(known_labels_noisy, label_enc_weight)
        else:
            input_query_content = self.label_enc(known_labels_noisy)

        # Position query: Box embedding (inverse sigmoid is usually done in the model, here we pass box)
        # We need to map 4 coords to hidden_dim. Usually MLP or Sine.
        # We will assume the caller handles the 4->Hidden mapping or we return 4 coords.
        # To match standard DINO, we usually pass the raw boxes as reference points (pos queries).
        # But the transformer expects (hidden_dim).
        # We will return the boxes, and the model wrapper will convert them to sine embeddings.
        input_query_pos = known_boxes_noisy  # (Total_DN, 4)

        # 5. Attention Mask
        # We need to mask so that:
        # - Each DN group sees only itself.
        # - Matching queries (original) see only themselves.
        # - DN groups cannot see Matching queries.
        # - Matching queries cannot see DN queries.

        # Total queries = Total_DN + Num_Matching
        num_matching = self.num_queries
        total_queries = total_dn_queries + num_matching

        # Initialize mask with 0 (visible)
        attn_mask = torch.zeros(
            (total_queries, total_queries), device=device, dtype=torch.bool
        )

        # Mask out cross-group visibility
        # Structure: [DN_Group_0, DN_Group_1, ..., Matching]
        # Each DN group has size `num_gt` (variable per batch, but here we flattened)
        # Wait, standard attention mask in PyTorch is (L, L) or (B*H, L, L).
        # Since num_gt varies per image, we usually pad or do this per batch item.
        # But here we flattened everything. This is tricky for batch processing.
        # DINO usually constructs a mask of size (Batch*Total_Q, Batch*Total_Q) or (Total_Q, Total_Q) if fixed.

        # Simplification for 24h task:
        # We will construct the mask for a single batch item logic, but expanded.
        # Actually, if we pad `known_labels` to a max size, it's easier.
        # But we used flatten.
        # To make this work with standard MultiheadAttention (which expects (L, B, C)),
        # we need to pad the DN queries to a fixed size per batch or handle variable lengths.

        # Let's assume we pad to a fixed `max_gt` for simplicity in this implementation,
        # or we just return the flattened version and let the model handle the reshaping/padding.
        # Given the constraints, we will return the flattened tensors and a `dn_meta` dict
        # that tells the model how to split them.

        # However, to generate a valid mask for the Transformer, we really need the tensor to be (L, B, C).
        # We will reshape `input_query_content` to (num_patterns*2*max_gt, B, C).

        # Re-organizing to (Num_DN_Groups * Max_GT, B, C)
        # Find max_gt in this batch
        max_gt = 0
        for t in targets:
            max_gt = max(max_gt, len(t["labels"]))

        if max_gt == 0:
            return None, None, None, None

        # Pad and stack
        padded_content = []
        padded_boxes = []

        # We iterate over patterns
        # For each pattern, for each batch item...

        # This is getting complex. Let's simplify:
        # We return the raw noisy targets and let the training loop handle the masking/padding?
        # No, the prompt asks for `ContrastiveDeNoising` logic.

        # Let's produce the mask for the self-attention.
        # We will assume the output tensor to the transformer will be:
        # [DN_Queries (P * 2 * MaxGT), Matching_Queries (N), Batch] -> No, usually (L, B, C).
        # L = P * 2 * MaxGT + N.

        L_dn = self.num_patterns * 2 * max_gt
        L_all = L_dn + num_matching

        attn_mask = torch.ones(
            (L_all, L_all), device=device, dtype=torch.bool
        )  # 1 means ignore (blocked)

        # Allow matching queries to see each other
        attn_mask[L_dn:, L_dn:] = False

        # Allow each DN group to see itself
        for i in range(self.num_patterns * 2):
            start = i * max_gt
            end = (i + 1) * max_gt
            # Unmask this block
            attn_mask[start:end, start:end] = False

        # 6. Prepare Output Tensors (L_dn, B, C)
        # We need to fill the padded structure
        dn_content = torch.zeros((L_dn, batch_size, self.hidden_dim), device=device)
        dn_boxes = torch.zeros((L_dn, batch_size, 4), device=device)

        # Fill
        # We generated flattened noisy data earlier. We need to map it back to (Group, Batch, GT).
        # It's easier to regenerate with padding logic.

        idx_ptr = 0
        for p in range(self.num_patterns * 2):
            # Determine noise parameters for this group
            is_neg = p >= self.num_patterns
            this_box_scale = (
                self.box_noise_scale * 2.0 if is_neg else self.box_noise_scale
            )

            for b in range(batch_size):
                labels = targets[b]["labels"]
                boxes = targets[b]["boxes"]
                n_gt = len(labels)

                if n_gt > 0:
                    # Noise
                    noise_box = torch.rand_like(boxes) * 2 - 1
                    diff = torch.zeros_like(boxes)
                    diff[:, :2] = boxes[:, 2:] * 0.5
                    diff[:, 2:] = boxes[:, 2:] * 0.5
                    noisy_boxes = boxes + noise_box * diff * this_box_scale
                    noisy_boxes = noisy_boxes.clamp(0, 1)

                    noisy_labels = labels.clone()
                    if (
                        self.label_noise_scale > 0 and is_neg
                    ):  # Apply label noise mostly to neg
                        if torch.rand(1).item() < self.label_noise_scale:
                            noisy_labels = torch.randint_like(
                                labels, 0, self.num_classes
                            )

                    # Embed
                    if label_enc_weight is not None:
                        c_emb = F.embedding(noisy_labels, label_enc_weight)
                    else:
                        c_emb = self.label_enc(noisy_labels)

                    # Place in tensor
                    start_idx = p * max_gt
                    dn_content[start_idx : start_idx + n_gt, b, :] = c_emb
                    dn_boxes[start_idx : start_idx + n_gt, b, :] = noisy_boxes

        dn_meta = {
            "pad_size": max_gt,
            "num_dn_group": self.num_patterns * 2,
            "L_dn": L_dn,
        }

        return dn_content, dn_boxes, attn_mask, dn_meta


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
