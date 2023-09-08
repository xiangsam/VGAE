from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./datasets', name='Cora')[0]
print(dataset)