import torch

class StandardNorm(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 2e-12)
        return x