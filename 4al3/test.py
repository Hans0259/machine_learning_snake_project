import torch
print("Torch:", torch.__version__)
print("CUDA in torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
