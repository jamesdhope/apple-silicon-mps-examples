import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Setup the distributed process group."""
    # Use 'gloo' for CPU backend
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Process {rank} initialized.")

def cleanup():
    """Cleanup the distributed process group."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """A simple linear model."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def train(rank, world_size):
    """Training loop for a single process."""
    setup(rank, world_size)

    # Initialize model, optimizer, and DDP
    model = SimpleModel()
    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy dataset
    data = torch.randn(100, 10)
    targets = torch.randn(100, 1)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train(rank, world_size)
