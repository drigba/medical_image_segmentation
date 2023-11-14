class ResnetLoss:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, input, target):
        input = input["out"]
        return self.loss(input,target)