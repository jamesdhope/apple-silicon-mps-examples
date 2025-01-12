import torch
print(torch.distributed.is_available())
print(torch.distributed.is_gloo_available())  # Should be True for gloo backend support
print(torch.distributed.is_nccl_available())  # Should be False on macOS as it needs GPUs
print(torch.distributed.is_mpi_available())  # Should be False on macOS unless mpi is installed
