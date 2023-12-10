import torch

class MultiUnet(torch.nn.Module):
    """
    MultiUnet is a class that represents a multi-branch U-Net model.

    Args:
        model_list (list): A list of feature models for each branch.
        prediction_model (torch.nn.Module): The prediction model for the final output.

    Attributes:
        feature_models (list): A list of feature models for each branch.
        prediction_model (torch.nn.Module): The prediction model for the final output.
    """

    def __init__(self, model_list, prediction_model):
        super().__init__()
        self.feature_models = model_list
        self.prediction_model = prediction_model
        
    def forward(self, x):
        """
        Forward pass of the MultiUnet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        outputs = [model(x) for model in self.feature_models]
        outputs = torch.cat(outputs, dim=1)
        return self.prediction_model(outputs)