import torch

class DivergentNets(torch.nn.Module):
    def __init__(self, model_list):
        """
        Initializes a DivergentNets instance.

        Args:
            model_list (list): A list of models to be used in the ensemble.

        """
        super().__init__()
        self.model_list = model_list

    def forward(self, x):
        """
        Performs forward pass through the ensemble of models.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the ensemble prediction.

        """
        outputs = [model(x) for model in self.model_list]
        outputs = torch.tensor(outputs)
        prediction = torch.mean(outputs, dim=0)
        return prediction