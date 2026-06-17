# -*- coding: utf-8 -*-
"""Vision Transformer (ViT) backbone for deepfake detection.

Architecture Overview
---------------------
We leverage ViT-Base/16 (patch size 16, 12 transformer layers, 768-dim
embeddings) pretrained on ImageNet-21k and fine-tuned on ImageNet-1k via
the ``timm`` library.  The standard classification token is fed through a
custom head (LayerNorm → Linear) to produce binary logits.

A key advantage of ViT for deepfake detection is the self-attention
mechanism: it can capture long-range spatial dependencies (e.g., inconsistent
lighting between face and background) that local convolutions may miss.
We expose per-layer attention weights so that they can be visualised as
"attention rollout" heatmaps.

Input:  (B, 3, 224, 224)  –  RGB face crops, ImageNet-normalised
Output: (B, num_classes)   –  logits (default num_classes=2: real / fake)

References
----------
* Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
* Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

__all__ = ["ViTDetector", "create_vit"]


class ViTDetector(nn.Module):
    """Vision Transformer fine-tuned for binary deepfake classification.

    Parameters
    ----------
    pretrained : bool
        If ``True``, load ImageNet-pretrained weights for the backbone.
    num_classes : int
        Number of output classes (default ``2``).
    dropout_rate : float
        Dropout probability in the classification head.

    Attributes
    ----------
    backbone : nn.Module
        ViT-Base/16 without the original classification head.
    head : nn.Sequential
        LayerNorm → Dropout → Linear producing final logits.
    attention_weights : list[torch.Tensor]
        Populated after a forward pass when :pymethod:`enable_attention_hooks`
        has been called.  Each element is shape ``(B, num_heads, N, N)`` where
        ``N = 1 + (H/P)*(W/P)`` (CLS token + patch tokens).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone – strip the built-in head so we get the CLS embedding.
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # removes head, forward returns CLS embedding
        )

        # Embedding dimension for ViT-Base
        self.embed_dim: int = 768

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.embed_dim, num_classes),
        )

        # ------ Attention weight storage ------
        self.attention_weights: list[torch.Tensor] = []
        self._attn_hooks: list = []

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------
    def enable_attention_hooks(self) -> None:
        """Register forward hooks on every transformer block to capture
        the raw attention weight matrices.

        After a forward pass, ``self.attention_weights`` will contain one
        tensor per layer, each of shape ``(B, num_heads, N, N)``.
        """
        self.attention_weights.clear()

        for block in self.backbone.blocks:
            hook = block.attn.register_forward_hook(self._capture_attention)
            self._attn_hooks.append(hook)

    def disable_attention_hooks(self) -> None:
        """Remove attention hooks and clear stored weights."""
        for hook in self._attn_hooks:
            hook.remove()
        self._attn_hooks.clear()
        self.attention_weights.clear()

    def _capture_attention(
        self,
        module: nn.Module,
        inp: tuple,
        out: torch.Tensor,
    ) -> None:
        """Hook callback that stores the attention matrix.

        timm's ``Attention`` module exposes ``self.attn_drop`` which is
        applied *after* softmax.  We grab the attention map by temporarily
        enabling ``attn.fused_attn = False`` and reading ``module.attn_weights``
        if available, or by recomputing from Q, K.
        """
        # timm ≥ 0.9 stores attention weights when `attn.fused_attn` is False.
        # As a robust fallback we recompute from the input.
        x = inp[0]  # (B, N, C)
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, _ = qkv.unbind(0)

        scale = (C // module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        self.attention_weights.append(attn.detach())

    def get_attention_rollout(self) -> Optional[torch.Tensor]:
        """Compute attention rollout across all layers.

        Attention rollout multiplies attention matrices layer-by-layer
        (adding an identity residual at each step) to approximate how
        information flows from input patches to the CLS token.

        Returns
        -------
        torch.Tensor or None
            Shape ``(B, N)`` giving the rollout attention from each patch
            to the CLS token, or ``None`` if no attention weights are stored.
        """
        if not self.attention_weights:
            return None

        # Average over heads, then multiply through layers with residual
        rollout: Optional[torch.Tensor] = None
        for attn in self.attention_weights:
            # attn: (B, heads, N, N)
            attn_mean = attn.mean(dim=1)                          # (B, N, N)
            identity = torch.eye(
                attn_mean.size(-1),
                device=attn_mean.device,
                dtype=attn_mean.dtype,
            ).unsqueeze(0)
            attn_with_residual = 0.5 * attn_mean + 0.5 * identity  # (B, N, N)

            if rollout is None:
                rollout = attn_with_residual
            else:
                rollout = rollout @ attn_with_residual              # (B, N, N)

        assert rollout is not None
        # Extract CLS row → attention to every patch
        cls_attention = rollout[:, 0, 1:]                           # (B, num_patches)
        # Normalise to [0, 1] per sample
        cls_attention = cls_attention / (cls_attention.sum(dim=-1, keepdim=True) + 1e-8)
        return cls_attention

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of face crops, shape ``(B, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(B, num_classes)``.
        """
        cls_embedding = self.backbone(x)   # (B, 768)
        logits = self.head(cls_embedding)  # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS embedding (useful for ensemble fusion).

        Parameters
        ----------
        x : torch.Tensor
            Batch of face crops, shape ``(B, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            CLS token embedding, shape ``(B, 768)``.
        """
        return self.backbone(x)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def create_vit(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.1,
) -> ViTDetector:
    """Instantiate a ViT-Base/16 deepfake detector.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained backbone weights.
    num_classes : int
        Number of output classes (``2`` for real / fake).
    dropout_rate : float
        Dropout probability in the classification head.

    Returns
    -------
    ViTDetector
        Ready-to-use model.

    Example
    -------
    >>> model = create_vit(pretrained=True, num_classes=2)
    >>> dummy = torch.randn(1, 3, 224, 224)
    >>> logits = model(dummy)
    >>> logits.shape
    torch.Size([1, 2])
    """
    return ViTDetector(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
