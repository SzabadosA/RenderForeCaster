import torch

print("Is CUDA available? ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("Number of CUDA devices: ", torch.cuda.device_count())
print("Current CUDA device: ", torch.cuda.current_device())
print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
