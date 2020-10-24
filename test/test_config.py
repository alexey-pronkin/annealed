import torch
DEVICE = 'cpu'
device = DEVICE if DEVICE else ('cuda' if torch.cuda.is_available() else 'cpu')
gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu_device0 = 'cpu:0'
cpu_device1 = 'cpu:1'