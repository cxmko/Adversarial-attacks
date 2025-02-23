import torch
from torch import nn
import torch.nn.functional as F

class FastGradientSignMethod:
    """
    Implements the Fast Gradient Sign Method (FGSM) attack for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
    """
    def __init__(self, model, eps):
        """
        Initializes the FGSM attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
        """
        self.model = model
        self.eps = eps
        self.name = f'FGSM_{eps:.2e}'

    def compute(self, x, y):
        """
        Generates an adversarial perturbation using FGSM.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

         Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)

        # get model output and compute loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(self.model(x + delta), y)
        loss.backward()

        ## apply one step of sign gradient ascent to the input

        perturbation = self.eps * delta.grad.detach().sign()
        return perturbation

class ProjectedGradientDescent:
    """
    Implements the Projected Gradient Descent (PGD) attack in L-infinity norm for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
        alpha (float): Step size for each iteration.
        num_iter (int): Number of iterations for the attack.
    """
    def __init__(self, model, eps, alpha, num_iter):
        """
        Initializes the PGD attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
            alpha (float): Step size for the attack.
            num_iter (int): Number of attack iterations.
        """
        self.model = model
        self.eps = eps
        self.num_iter = num_iter

        ## To do 19
        self.alpha = self.eps / self.num_iter
        self.name = f'PGDLinf_{eps:.2e}_{alpha:.2e}_{num_iter}'


    def compute(self, x, y):
        """
        Generates an adversarial perturbation using PGD with L2 norm.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

        Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)

        # iteratively compute adversarial perturbations
        for t in range(self.num_iter):
            # get model output and compute loss (cross-entropy)
            output = self.model(x + delta)
            loss = F.cross_entropy(output, y)
            
            loss.backward()
            
            grad = delta.grad.detach()
            delta.data = delta + self.alpha * grad.sign()

            delta.data = torch.clamp(delta, -self.eps, self.eps)            
            delta.grad.zero_()
        
        return delta.detach()
