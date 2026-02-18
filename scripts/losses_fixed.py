#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
losses.py - Custom loss functions for CT report classification

Improvements:
- Added comprehensive type hints
- Moved eff() to CBLoss as static method
- Added docstrings for all classes and methods
- Enhanced numerical stability
- Fixed device-related issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard examples.
    
    FL(pt) = -(1-pt)^γ * log(pt)
    
    Args:
        gamma: Focusing parameter. Higher gamma puts more focus on hard examples.
        class_weights: Optional class weights for additional balancing.
    """
    
    def __init__(self, gamma: float = 1.0, class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Ensure targets are on the same device as logits
        targets = targets.to(logits.device)
        
        # Get class weights on the correct device
        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        
        ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)  # Improved numerical stability
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


class CBLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples.
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    Args:
        counts: List or tensor of sample counts per class
        beta: Hyperparameter for re-weighting. Typically 0.9999 (IB-Loss experiment)
        gamma: Optional focal parameter for CB-Focal variant
    """
    
    def __init__(self, counts: Union[list, torch.Tensor], beta: float = 0.9999, gamma: float = 0.0) -> None:
        super().__init__()
        if isinstance(counts, list):
            counts = torch.tensor(counts, dtype=torch.float)
        
        # Calculate effective number of samples
        effective_num = self.calculate_effective_num(beta, counts)
        
        # Calculate weights
        weights = (1 - beta) / effective_num
        weights = weights / weights.sum() * len(counts)
        
        self.register_buffer("weights", weights)
        self.gamma = gamma
    
    @staticmethod
    def calculate_effective_num(beta: float, counts: torch.Tensor) -> torch.Tensor:
        """
        Calculate effective number of samples for each class.
        
        Args:
            beta: Re-weighting hyperparameter (0 < beta < 1)
            counts: Number of samples per class
        
        Returns:
            Effective number of samples per class
        """
        return (1 - beta ** counts) / (1 - beta)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Ensure targets are on the same device as logits
        targets = targets.to(logits.device)
        
        # Ensure weights are on the correct device
        weights = self.weights.to(logits.device)
        
        log_probs = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_probs, targets, reduction="none", weight=weights)
        
        if self.gamma > 0:
            # CB-Focal variant
            probs = torch.exp(log_probs)
            pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
            pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
            focal_weight = (1 - pt) ** self.gamma
            ce = focal_weight * ce
        
        return ce.mean()


class IBLoss(nn.Module):
    """
    Influence-Balanced Loss implementation.
    
    Computes IB weights as: grads * features, where grads = ||softmax(logits) - one_hot(target)||₁
    
    Args:
        weight: Class weights for balancing
        alpha: Scaling parameter for influence weights (default: 1000.0, IB-Loss experiment)
        gamma: Optional focal parameter for IB-Focal variant
        cb_counts: Optional sample counts per class for CB weighting (IB-CB variant)
        cb_beta: Beta parameter for CB weighting (default: 0.9999, IB-Loss experiment)
    """
    
    def __init__(self, weight=None, alpha=1000., gamma=None, cb_counts=None, cb_beta=0.9999):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma
        
        # CB weighting setup
        if cb_counts is not None:
            if isinstance(cb_counts, list):
                cb_counts = torch.tensor(cb_counts, dtype=torch.float)
            
            # Calculate CB weights using the same method as CBLoss
            effective_num = CBLoss.calculate_effective_num(cb_beta, cb_counts)
            cb_weights = (1 - cb_beta) / effective_num
            cb_weights = cb_weights / cb_weights.sum() * len(cb_counts)
            self.register_buffer('cb_weights', cb_weights)
        else:
            self.register_buffer('cb_weights', None)
    
    def forward(self, input, target, features):
        num_classes = input.size(-1)
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1) # N * 1
        
        # Compute feature magnitude per sample (L1 norm)
        feature_norms = torch.sum(torch.abs(features), dim=1)  # N
        
        # Compute IB weights
        ib = grads * feature_norms
        ib = self.alpha / (ib + self.epsilon)
        
        # Determine class weights (CB weights take precedence over manual weights)
        class_weights = self.weight
        if self.cb_weights is not None:
            class_weights = self.cb_weights.to(input.device)
        
        # Base cross-entropy loss with CB weighting
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=class_weights)
        
        # Apply focal weighting if specified (only when CB is not used)
        if self.gamma is not None and self.gamma > 0 and self.cb_weights is None:
            with torch.no_grad():
                pt = torch.exp(-ce_loss)
                pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
                focal_weight = (1 - pt) ** self.gamma
            ce_loss = ce_loss * focal_weight
        
        # Apply IB weighting and return mean loss
        loss = ce_loss * ib
        return loss.mean()


