#pip install torch torchvision torchaudio
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Use GPU
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    # Use CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Example usage of GPU
x = torch.randn(3, 3).to(device)
