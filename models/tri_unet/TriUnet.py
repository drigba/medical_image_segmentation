import torch

class TriUnet(torch.nn.Module):
    def __init__(self, model_list, prediction_model):
        super().__init__()
        self.feature_models = model_list
        self.prediction_model = prediction_model
        
    def forward(self, x):
        outputs = [model(x) for model in self.feature_models]
        outputs = torch.cat(outputs, dim=1)
        return self.prediction_model(outputs)