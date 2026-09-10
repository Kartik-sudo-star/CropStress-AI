"""
Grad-CAM and Explainability for Deepfake Detection

Implements Grad-CAM, Grad-CAM++, Integrated Gradients, and other
visualization methods for model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple, Callable
import cv2
import logging
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    Occlusion,
    NoiseTunnel,
)
from captum.attr import visualization as viz

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for CNN-based models.
    
    Generates class activation maps showing which regions
    influenced the model's prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = False
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: [1, C, H, W] or [B, C, H, W]
            target_class: Target class index (None = predicted class)
            
        Returns:
            heatmap: [H, W] numpy array in range [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, target_class] if output.dim() > 1 else output[target_class]
        target.backward(retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # [B, C, H, W] or [B, C, H, W]
        gradients = self.gradients      # [B, C, H, W] or [B, C, H, W]
        
        if activations is None or gradients is None:
            logger.warning("No activations/gradients captured. Check target layer.")
            return np.zeros((224, 224))
        
        # Use first sample if batch
        if activations.dim() == 4:
            activations = activations[0]  # [C, H, W]
            gradients = gradients[0]      # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        input_size = input_tensor.shape[-2:]
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        return cam
    
    def __del__(self):
        try:
            self.remove_hooks()
        except:
            pass


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation with improved weighting."""
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)
        
        activations = self.activations
        gradients = self.gradients
        
        if activations is None or gradients is None:
            return np.zeros((224, 224))
        
        if activations.dim() == 4:
            activations = activations[0]
            gradients = gradients[0]
        
        # Grad-CAM++ weights
        # alpha = grad^2 / (2*grad^2 + sum(act * grad^3))
        grad_sq = gradients ** 2
        grad_cu = gradients ** 3
        
        sum_act_grad_cu = (activations * grad_cu).sum(dim=(1, 2), keepdim=True)
        denom = 2 * grad_sq + sum_act_grad_cu + 1e-8
        alphas = grad_sq / denom
        
        weights = (alphas * F.relu(gradients)).sum(dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        input_size = input_tensor.shape[-2:]
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        return cam


class IntegratedGradientsExplainer:
    """Integrated Gradients for attribution."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.ig = IntegratedGradients(model)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution.
        
        Args:
            input_tensor: [1, C, H, W] or [B, C, H, W]
            target_class: Target class index
            baseline: Baseline tensor (zeros if None)
            steps: Number of interpolation steps
            
        Returns:
            attribution: [H, W] or [C, H, W] numpy array
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_tensor).argmax(dim=1).item()
        
        # Compute attributions
        attributions, delta = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=steps,
            return_convergence_delta=True
        )
        
        # Sum over channels for visualization
        if attributions.dim() == 4:
            attr = attributions[0].sum(dim=0).cpu().numpy()
        else:
            attr = attributions.sum(dim=0).cpu().numpy()
        
        # Normalize
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
        
        return attr


class OcclusionExplainer:
    """Occlusion-based sensitivity analysis."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.occlusion = Occlusion(model)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        sliding_window_shapes: Tuple[int, int, int] = (3, 15, 15),
        strides: Tuple[int, int, int] = (3, 8, 8),
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate occlusion sensitivity map.
        
        Args:
            input_tensor: [1, C, H, W]
            target_class: Target class index
            sliding_window_shapes: (C, H, W) occlusion window
            strides: (C, H, W) strides
            baseline: Baseline value for occlusion
            
        Returns:
            sensitivity_map: [H, W] numpy array
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_tensor).argmax(dim=1).item()
        
        if baseline is None:
            baseline = 0.0
        
        attributions = self.occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baseline
        )
        
        # Sum over channels
        if attributions.dim() == 4:
            attr = attributions[0].sum(dim=0).cpu().numpy()
        else:
            attr = attributions.sum(dim=0).cpu().numpy()
        
        # Normalize (higher = more important)
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
        
        return attr


class VideoGradCAM:
    """Grad-CAM for video models (per-frame)."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = False
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __call__(
        self,
        video_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        frame_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM for each frame in video.
        
        Args:
            video_tensor: [1, T, C, H, W] or [B, T, C, H, W]
            target_class: Target class index
            frame_indices: Specific frames to analyze (None = all)
            
        Returns:
            List of heatmaps [H, W] for each frame
        """
        self.model.eval()
        video_tensor = video_tensor.to(self.device)
        
        if video_tensor.dim() == 5:
            video_tensor = video_tensor[0]  # [T, C, H, W]
        
        T = video_tensor.shape[0]
        
        if frame_indices is None:
            frame_indices = list(range(T))
        
        heatmaps = []
        
        for t in frame_indices:
            if t >= T:
                continue
                
            frame = video_tensor[t:t+1]  # [1, C, H, W]
            
            # Forward pass through frame backbone
            # We need to extract features for single frame
            with torch.enable_grad():
                frame.requires_grad_(True)
                
                # Get frame features
                if hasattr(self.model, 'extract_frame_features'):
                    features = self.model.extract_frame_features(frame.unsqueeze(0))
                    logits = self.model.classifier(features.squeeze(1))
                else:
                    logits = self.model(frame)
                
                if target_class is None:
                    target_class = logits.argmax(dim=1).item()
                
                self.model.zero_grad()
                target = logits[0, target_class]
                target.backward(retain_graph=True)
                
                activations = self.activations
                gradients = self.gradients
                
                if activations is not None and gradients is not None:
                    if activations.dim() == 4:
                        activations = activations[0]
                        gradients = gradients[0]
                    
                    weights = gradients.mean(dim=(1, 2))
                    cam = torch.zeros(activations.shape[1:], device=self.device)
                    for i, w in enumerate(weights):
                        cam += w * activations[i]
                    
                    cam = F.relu(cam)
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    
                    cam = cam.cpu().numpy()
                    input_size = frame.shape[-2:]
                    cam = cv2.resize(cam, (input_size[1], input_size[0]))
                    heatmaps.append(cam)
                else:
                    heatmaps.append(np.zeros((224, 224)))
        
        return heatmaps
    
    def __del__(self):
        try:
            self.remove_hooks()
        except:
            pass


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: [H, W, 3] RGB image (0-255 or 0-1)
        heatmap: [H, W] heatmap (0-1)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        overlay: [H, W, 3] RGB image
    """
    # Ensure image is 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


def create_explanation_figure(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create side-by-side explanation figure.
    
    Returns:
        Combined image [H, 3*W, 3] with original, heatmap, overlay
    """
    overlay = overlay_heatmap(original_image, heatmap, alpha, colormap)
    
    # Ensure all same size
    h, w = original_image.shape[:2]
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Combine horizontally
    combined = np.hstack([
        original_image if original_image.max() > 1 else (original_image * 255).astype(np.uint8),
        heatmap_color,
        overlay
    ])
    
    return combined


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to numpy image for visualization."""
    # Denormalize ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = tensor.clone().detach().cpu()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img