import torch

class DivergentNets(torch.nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = model_list

    def forward(self, x):
        outputs = [model(x) for model in self.model_list]
        outputs = torch.tensor(outputs)
        prediction = torch.mean(outputs, dim=0)
        return prediction