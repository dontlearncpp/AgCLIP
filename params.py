import torch
import torchvision
from models_counting_network import CountingNetwork

# Model
print('==> Building model..')
model = CountingNetwork()



pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total - ', pytorch_total_params)
print('Trainable - ', trainable_pytorch_total_params)
