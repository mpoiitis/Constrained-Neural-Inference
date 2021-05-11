import torch
import random
import numpy as np
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from utils import seed_worker

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

train_dataset = ZINC(root='data/ZINC')
test_dataset = ZINC(root='data/ZINC', split='test')
val_dataset = ZINC(root='data/ZINC', split='val')

print('Graphs for each split    Train:{}, Test:{}, Val:{}'.format(len(train_dataset), len(test_dataset), len(val_dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker)