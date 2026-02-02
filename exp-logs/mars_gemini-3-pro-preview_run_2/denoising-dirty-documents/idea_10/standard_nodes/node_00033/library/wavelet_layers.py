import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    """
    Discrete Wavelet Transform (DWT) Layer using Haar Wavelet.
    Performs downsampling by decomposing the input into frequency subbands.
    """

    def __init__(self):
        super(DWT, self).__init__()
        # DWT is a fixed mathematical operation, no learnable parameters.
        self.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Decomposed tensor of shape (N, 4C, H/2, W/2).
                          Channels are ordered: [LL, LH, HL, HH].
        """
        # Ensure dimensions are even for 2x2 block processing
        h, w = x.shape[2], x.shape[3]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Slice the input into 2x2 blocks
        # x00: Top-Left, x01: Top-Right
        # x10: Bottom-Left, x11: Bottom-Right
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # Haar Wavelet Transform formulas
        # We use a scaling factor of 0.5 to ensure energy conservation/invertibility
        # consistent with the IWT implementation below.

        # LL: Low-Low (Approximation) - Average
        LL = 0.5 * (x00 + x01 + x10 + x11)

        # LH: Low-High (Vertical Detail) - Low pass on Y, High pass on X
        # Note: Convention varies, here we define based on standard Haar matrix application
        # LH captures vertical structures (diff between columns is small, diff between rows is large?
        # No, usually LH means Low freq on dim 0 (Y), High freq on dim 1 (X)).
        # Formula: (x00 - x01) + (x10 - x11) -> Sum of row differences
        LH = 0.5 * (x00 - x01 + x10 - x11)

        # HL: High-Low (Horizontal Detail) - High pass on Y, Low pass on X
        # Formula: (x00 + x01) - (x10 + x11) -> Diff of row sums
        HL = 0.5 * (x00 + x01 - x10 - x11)

        # HH: High-High (Diagonal Detail) - High pass on Y, High pass on X
        # Formula: (x00 - x01) - (x10 - x11) -> Diff of row differences
        HH = 0.5 * (x00 - x01 - x10 + x11)

        # Concatenate along channel dimension
        return torch.cat([LL, LH, HL, HH], dim=1)


class IWT(nn.Module):
    """
    Inverse Discrete Wavelet Transform (IWT) Layer using Haar Wavelet.
    Performs upsampling by reconstructing the signal from frequency subbands.
    """

    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, 4C, H, W).
                              Expected channel order: [LL, LH, HL, HH].

        Returns:
            torch.Tensor: Reconstructed tensor of shape (N, C, 2H, 2W).
        """
        # Split channels back into subbands
        # x shape: (N, 4*C_out, H, W)
        r_channel = x.size(1) // 4
        LL, LH, HL, HH = torch.split(x, r_channel, dim=1)

        # Inverse Haar Transform formulas
        # Derived by solving the system of linear equations from DWT

        # x00 = 0.5 * (LL + LH + HL + HH)
        x00 = 0.5 * (LL + LH + HL + HH)

        # x01 = 0.5 * (LL - LH + HL - HH)
        x01 = 0.5 * (LL - LH + HL - HH)

        # x10 = 0.5 * (LL + LH - HL - HH)
        x10 = 0.5 * (LL + LH - HL - HH)

        # x11 = 0.5 * (LL - LH - HL + HH)
        x11 = 0.5 * (LL - LH - HL + HH)

        # Reconstruct the spatial tensor by interleaving pixels
        # Output shape: (N, C_out, 2H, 2W)
        out_batch, out_channel, in_height, in_width = LL.size()
        out_height, out_width = in_height * 2, in_width * 2

        output = torch.zeros(
            (out_batch, out_channel, out_height, out_width),
            dtype=x.dtype,
            device=x.device,
        )

        output[:, :, 0::2, 0::2] = x00
        output[:, :, 0::2, 1::2] = x01
        output[:, :, 1::2, 0::2] = x10
        output[:, :, 1::2, 1::2] = x11

        return output
