import torch
import os
torch.manual_seed(0)

deviceids = [4, 5]

class MockModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.layer = torch.nn.Parameter(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float))
	
	def forward(self, x):
		return torch.matmul(x, self.layer)

# Setup the multi-GPU training
from torch.distributed import init_process_group, destroy_process_group

# Check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1
# if ddp than setup per process control variables
if ddp:
	print('ddp available')
	assert torch.cuda.is_available()
	init_process_group(backend='nccl')
	ddp_rank = int(os.environ['RANK'])
	ddp_local_rank = int(os.environ['LOCAL_RANK'])
	ddp_world_size = int(os.environ['WORLD_SIZE'])
	assert ddp_world_size == len(deviceids) # for now just to make sure
	device = f'cuda:{deviceids[ddp_rank]}'
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0
else:
	print('no ddp')
	exit(-1)

# Create the model
mm = MockModel()
mm.to(device)

train_mm = torch.nn.parallel.DistributedDataParallel(mm, device_ids=[deviceids[ddp_rank]])

data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
targets = torch.tensor([0, 1])

local_micro_batch = data[ddp_rank].view(1, -1).to(device)

logits = train_mm(local_micro_batch)

loss = torch.nn.functional.cross_entropy(logits, targets[ddp_rank].view(1).to(device))

loss.backward()

print(f'ddp_rank {ddp_rank} \n {train_mm.module.layer.grad}')