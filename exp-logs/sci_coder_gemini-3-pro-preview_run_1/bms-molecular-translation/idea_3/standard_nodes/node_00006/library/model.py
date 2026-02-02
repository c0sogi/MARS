import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
from library.config import Config


class Encoder(nn.Module):
    """
    ResNet-18 based encoder that outputs a sequence of feature vectors
    representing the spatial grid of the image.
    """

    def __init__(self, d_model=256):
        super().__init__()
        # Load pretrained ResNet18
        # Using the modern 'weights' parameter if available, else fallback to pretrained=True
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            resnet = torchvision.models.resnet18(weights=weights)
        except (AttributeError, ImportError):
            resnet = torchvision.models.resnet18(pretrained=True)

        # Remove avgpool and fc layers to keep spatial features
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        # Output of layer4 is (Batch, 512, H/32, W/32)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Project 512 channels to d_model
        self.projection = nn.Conv2d(512, d_model, kernel_size=1)

        # Positional embedding for the flattened spatial grid
        # For 224x224 image, feature map is 7x7 = 49 patches
        self.grid_size = 7
        self.num_patches = self.grid_size * self.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(
            self.projection.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, images):
        # images: (Batch, 3, 224, 224)
        features = self.backbone(images)  # (Batch, 512, 7, 7)
        features = self.projection(features)  # (Batch, d_model, 7, 7)

        # Flatten spatial dimensions: (Batch, d_model, 49)
        features = features.flatten(2)

        # Permute to (Batch, 49, d_model) for Transformer
        features = features.permute(0, 2, 1)

        # Add positional embedding
        features = features + self.pos_embed

        return features


class Decoder(nn.Module):
    """
    Transformer Decoder that attends to image features and generates text sequence.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_len=410,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        # Learnable positional embedding for text
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        """
        tgt: (Batch, Seq_Len) - Input token indices
        memory: (Batch, Num_Patches, d_model) - Encoded image features
        tgt_mask: Causal mask
        tgt_pad_mask: Padding mask
        """
        seq_len = tgt.size(1)

        # Embeddings + Positional
        # We slice pos_embed to match current seq_len (useful during inference if step-by-step)
        # But usually in training seq_len is fixed max or batch max.
        # Safe slicing:
        pos = self.pos_embed[:, :seq_len, :]

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + pos
        tgt_emb = self.dropout(tgt_emb)

        # Transformer Decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        # Project to vocab size
        logits = self.fc_out(output)
        return logits


class VisualTransformer(nn.Module):
    """
    End-to-end CNN-Transformer model for Image Captioning (InChI prediction).
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = Encoder(d_model=Config.D_MODEL)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.N_HEAD,
            num_layers=Config.N_LAYER,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            max_len=Config.MAX_LEN,
        )

        self.vocab_size = vocab_size

    def make_masks(self, tgt, pad_idx):
        """
        Creates causal mask and padding mask.
        tgt: (Batch, Seq_Len)
        """
        seq_len = tgt.size(1)
        device = tgt.device

        # Causal mask: Square matrix where (i, j) is -inf if j > i
        # generate_square_subsequent_mask returns float mask with -inf/0
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Padding mask: (Batch, Seq_Len) boolean, True where padding
        tgt_pad_mask = tgt == pad_idx

        return tgt_mask, tgt_pad_mask

    def forward(self, images, text_seq, pad_idx=0):
        """
        Forward pass for training.
        images: (Batch, C, H, W)
        text_seq: (Batch, Seq_Len) - includes SOS, EOS, PAD
        pad_idx: int index of PAD token
        """
        # Encode images
        memory = self.encoder(images)

        # Create masks
        # For training, we feed the sequence excluding the last token (EOS/PAD) as input
        # and predict the sequence shifted by one (excluding SOS).
        # In PyTorch Transformer, we pass the input sequence `tgt`.
        # The caller usually handles the shifting (input vs target labels).
        # Here, we assume `text_seq` is the input to the decoder (e.g. SOS ... token_n).

        tgt_mask, tgt_pad_mask = self.make_masks(text_seq, pad_idx)

        # Decode
        logits = self.decoder(
            text_seq, memory, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask
        )

        return logits
