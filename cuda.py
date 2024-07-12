import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Set a random seed for CUDA tensors
    torch.cuda.manual_seed(123)
    
    # Example: Create a random CUDA tensor
    device = torch.device("cuda")
    tensor = torch.randn(3, 3, device=device)
    
    # Use the tensor for operations
    print(tensor)
else:
    print("CUDA is not available. Install CUDA-enabled PyTorch for GPU support.")
