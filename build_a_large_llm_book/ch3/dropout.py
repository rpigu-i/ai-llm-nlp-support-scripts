import torch

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
examples = torch.ones(6, 6)
print(dropout(examples))

