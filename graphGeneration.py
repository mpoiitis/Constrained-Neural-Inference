from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader


train_dataset = ZINC(root='data/ZINC')
test_dataset = ZINC(root='data/ZINC', split='test')
val_dataset = ZINC(root='data/ZINC', split='val')

print('Graphs for each split    Train:{}, Test:{}, Val:{}'.format(len(train_dataset), len(test_dataset), len(val_dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)