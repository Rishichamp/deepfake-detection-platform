# -*- coding: utf-8 -*-
"""EfficientNet-B4 backbone for deepfake detection.

Architecture Overview
---------------------
EfficientNet-B4 uses compound scaling (depth × width × resolution) to achieve
strong accuracy-per-FLOP.  We replace the original 1000-class ImageNet head
with a lightweight binary classifier and expose the final convolutional feature
map so that GradCAM can highlight the spatial regions most indicative of
manipulation artifacts (blending boundaries, texture inconsistencies, etc.).

Input:  (B, 3, 224, 224)  –  RGB face crops, ImageNet-normalised
Output: (B, num_classes)   –  logits (default num_classes=2: real / fake)

References
----------
* Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.
* Selvaraju et al., "Grad-CAM", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

__all__ = ["EfficientNetDetector", "create_efficientnet"]


class EfficientNetDetector(nn.Module):
    """EfficientNet-B4 fine-tuned for binary deepfake classification.

    Parameters
    ----------
    pretrained : bool
        If ``True``, load ImageNet-pretrained weights for the backbone.
    num_classes : int
        Number of output classes (default ``2`` – real vs fake).
    dropout_rate : float
        Dropout probability applied before the final linear layer.

    Attributes
    ----------
    backbone : nn.Module
        EfficientNet-B4 feature extractor (everything up to the global pool).
    global_pool : nn.AdaptiveAvgPool2d
        Spatial pooling layer producing a 1-D feature vector per sample.
    classifier : nn.Sequential
        Dropout → Linear head mapping pooled features to class logits.
    gradcam_features : torch.Tensor | None
        Populated after a forward pass when :pymethod:`enable_gradcam` has
        been called.  Contains the last convolutional feature map (B, C, H, W).
    gradcam_gradients : torch.Tensor | None
        Populated during the backward pass.  Contains the gradient of the
        loss w.r.t. ``gradcam_features``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Backbone – we create the full model then strip the classifier.
        # `features_only=False` gives us the whole model; we manually
        # separate the feature extractor from the head.
        # ------------------------------------------------------------------
        full_model = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # removes the built-in classifier
            global_pool="",         # removes the built-in pooling
        )
        self.backbone = full_model

        # Feature dimension of the last conv layer in EfficientNet-B4
        self.feature_dim: int = 1792

        # Spatial pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        # ------ GradCAM hooks (lazy-initialised) ------
        self.gradcam_features: Optional[torch.Tensor] = None
        self.gradcam_gradients: Optional[torch.Tensor] = None
        self._gradcam_hooks: list = []

    # ------------------------------------------------------------------
    # GradCAM support
    # ------------------------------------------------------------------
    def enable_gradcam(self) -> None:
        """Register forward/backward hooks to capture feature maps & grads.

        Call this *before* the forward pass you want to visualise. The
        captured tensors are stored in ``self.gradcam_features`` and
        ``self.gradcam_gradients`` respectively.
        """
        def _save_features(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            self.gradcam_features = out

        def _save_gradients(module: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
            self.gradcam_gradients = grad_out[0]

        # Hook into the *last* block of the backbone
        target_layer = self._get_gradcam_target_layer()
        self._gradcam_hooks.append(target_layer.register_forward_hook(_save_features))
        self._gradcam_hooks.append(target_layer.register_full_backward_hook(_save_gradients))

    def disable_gradcam(self) -> None:
        """Remove all GradCAM hooks and clear cached tensors."""
        for hook in self._gradcam_hooks:
            hook.remove()
        self._gradcam_hooks.clear()
        self.gradcam_features = None
        self.gradcam_gradients = None

    def _get_gradcam_target_layer(self) -> nn.Module:
        """Return the layer whose activations GradCAM should visualise.

        For EfficientNet-B4 (via timm) this is the last convolutional block,
        typically ``backbone.blocks[-1]``.
        """
        # timm's EfficientNet stores blocks as nn.Sequential in `.blocks`
        return self.backbone.blocks[-1]

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
        features = self.backbone(x)               # (B, 1792, 7, 7)
        pooled = self.global_pool(features)        # (B, 1792, 1, 1)
        pooled = pooled.flatten(1)                 # (B, 1792)
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector (useful for downstream tasks).

        Parameters
        ----------
        x : torch.Tensor
            Batch of face crops, shape ``(B, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Feature vector, shape ``(B, 1792)``.
        """
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        return pooled


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def create_efficientnet(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.3,
) -> EfficientNetDetector:
    """Instantiate an EfficientNet-B4 deepfake detector.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained backbone weights.
    num_classes : int
        Number of output classes (``2`` for real / fake).
    dropout_rate : float
        Dropout probability before the final linear layer.

    Returns
    -------
    EfficientNetDetector
        Ready-to-use model (call ``.to(device)`` and ``.train()`` / ``.eval()``).

    Example
    -------
    >>> model = create_efficientnet(pretrained=True, num_classes=2)
    >>> dummy = torch.randn(1, 3, 224, 224)
    >>> logits = model(dummy)
    >>> logits.shape
    torch.Size([1, 2])
    """
    return EfficientNetDetector(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
