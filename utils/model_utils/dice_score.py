import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculate the Dice coefficient between the input and target tensors.

    Args:
        input (Tensor): The input tensor.
        target (Tensor): The target tensor.
        reduce_batch_first (bool, optional): Whether to reduce the batch dimension first. Defaults to False.
        epsilon (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tensor: The Dice coefficient.

    Raises:
        AssertionError: If the input and target tensors have different sizes or dimensions.

    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculate the multiclass Dice coefficient.

    Args:
        input (Tensor): The predicted tensor.
        target (Tensor): The target tensor.
        reduce_batch_first (bool, optional): Whether to reduce the batch dimension first. Defaults to False.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        Tensor: The multiclass Dice coefficient.
    """
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """
    Calculates the dice loss between the predicted input and the target.

    Parameters:
        input (Tensor): The predicted input tensor.
        target (Tensor): The target tensor.
        multiclass (bool, optional): Whether to calculate the dice loss for multiclass segmentation. 
            Defaults to False.

    Returns:
        Tensor: The dice loss between the input and target tensors.
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
