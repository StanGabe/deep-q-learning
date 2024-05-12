import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

import torch
import torch.nn.functional as F

def discriminator_loss(logits_real, logits_fake):
    # Targets for real images are all ones, for fake images - all zeros
    true_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    
    # Loss for the real images
    loss_real = F.binary_cross_entropy_with_logits(logits_real, true_labels)
    # Loss for the fake images
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    
    # Total discriminator loss
    loss = loss_real + loss_fake
    return loss

def generator_loss(logits_fake):
    # The generator's goal is for the discriminator to mistake images as real
    true_labels = torch.ones_like(logits_fake)
    
    # Loss for what the discriminator outputs
    loss = F.binary_cross_entropy_with_logits(logits_fake, true_labels)
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    # Loss for the real images - how far from 1
    loss_real = F.mse_loss(scores_real, torch.ones_like(scores_real))
    # Loss for the fake images - how far from 0
    loss_fake = F.mse_loss(scores_fake, torch.zeros_like(scores_fake))
    
    # Total loss is the average of these two losses
    loss = 0.5 * (loss_real + loss_fake)
    return loss

def ls_generator_loss(scores_fake):
    # Generator wants discriminator to think the fakes are real (score of 1)
    loss = 0.5 * F.mse_loss(scores_fake, torch.ones_like(scores_fake))
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):  # support for scalar alpha
            self.alpha = torch.tensor([alpha, 1-alpha], dtype=torch.float32)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.size_average = size_average

    def forward(self, inputs, targets):
        # Ensure we're working with float32 for consistency
        inputs = inputs.type_as(targets)
        
        # Calculate the probability of being classified as class '1'
        probs = torch.sigmoid(inputs)
        # Calculate the binary cross entropy loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Here we adjust the targets to be either alpha or 1-alpha based on their class
        alpha_factor = torch.where(targets == 1, self.alpha[0], self.alpha[1])
        alpha_factor = alpha_factor.type_as(probs)

        # Modulate the loss with the focusing parameter
        focal_weight = torch.where(targets == 1, 1 - probs, probs)
        focal_weight = alpha_factor * (focal_weight ** self.gamma)

        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss

        # Average the loss if size_average is true
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def discriminator_loss_focal(logits_real, logits_fake, focal_loss_fn):
    """
    Compute the discriminator loss with focal loss component.
    """
    real_loss = focal_loss_fn(logits_real, torch.ones_like(logits_real))
    fake_loss = focal_loss_fn(logits_fake, torch.zeros_like(logits_fake))
    return real_loss + fake_loss

def generator_loss_focal(logits_fake, focal_loss_fn):
    """
    Compute the generator loss using the focal mechanism to fool the discriminator.
    """
    return focal_loss_fn(logits_fake, torch.ones_like(logits_fake))
