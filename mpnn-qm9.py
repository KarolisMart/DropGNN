# This implementation is based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py
# The evaluation setup is taken from https://github.com/chrsmrrs/k-gnn/blob/master/examples/1-qm9.py
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, degree


parser = argparse.ArgumentParser()
parser.add_argument('--target', default=0)
parser.add_argument('--drop_gnn', action='store_true', default=False)
parser.add_argument('--aux_loss', action='store_true', default=False)
args = parser.parse_args()
print(args)
target = int(args.target)
print('---- Target: {} ----'.format(target))
dim = 64

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0, keepdim=True)
std = dataset.data.y[tenpercent:].std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std

# Split datasets.
test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set the sampling probability and number of runs/samples for the DropGIN
n = []
degs = []
for g in dataset:
    num_nodes = g.num_nodes
    deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
    n.append(g.num_nodes)
    degs.append(deg.max())
print(f'Mean Degree: {torch.stack(degs).float().mean()}')
print(f'Max Degree: {torch.stack(degs).max()}')
print(f'Min Degree: {torch.stack(degs).min()}')
mean_n = torch.tensor(n).float().mean().round().long().item()
print(f'Mean number of nodes: {mean_n}')
print(f'Max number of nodes: {torch.tensor(n).float().max().round().long().item()}')
print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
print(f'Number of graphs: {len(dataset)}')
gamma = mean_n
p = 2 * 1 /(1+gamma)
num_runs = gamma
print(f'Number of runs: {num_runs}')
print(f'Sampling probability: {p}')
gamma = mean_n
p = 2 * 1 /(1+gamma)
num_runs = gamma
print(f'Number of runs: {num_runs}')
print(f'Sampling probability: {p}')

class MPNN(torch.nn.Module):
    def __init__(self):
        super(MPNN, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1), None

class DropMPNN(torch.nn.Module):
    def __init__(self):
        super(DropMPNN, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        if args.aux_loss:
            self.set2set_aux = Set2Set(dim, processing_steps=3)
            self.lin1_aux = torch.nn.Linear(2 * dim, dim)
            self.lin2_aux = torch.nn.Linear(dim, 1)

    def forward(self, data):
        aux_x = None
        # Do runs in paralel, by repeating the graphs in the batch
        x = data.x
        x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        x = x.view(-1, x.size(-1))
        run_edge_index = data.edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=data.edge_index.device).repeat_interleave(data.edge_index.size(1)) * (data.edge_index.max() + 1)
        run_edge_attr = data.edge_attr.repeat(num_runs, 1)
        x = F.relu(self.lin0(x))
        h = x.unsqueeze(0)
        for i in range(3):
            m = F.relu(self.conv(x, run_edge_index, run_edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)
        del run_edge_index, run_edge_attr
        if args.aux_loss:
            run_batch = data.batch.repeat(num_runs) + torch.arange(num_runs, device=data.batch.device).repeat_interleave(data.batch.size(0)) * (data.batch.max() + 1)
            aux_x = self.set2set_aux(x, run_batch)
            del run_batch
            aux_x = F.relu(self.lin1_aux(aux_x))
            aux_x = self.lin2_aux(aux_x)
            aux_x = aux_x.view(num_runs, -1, aux_x.size(-1))
        x = x.view(num_runs, -1, x.size(-1))
        x = x.mean(dim=0)

        x = self.set2set(x, data.batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1), aux_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_aux_loss = args.aux_loss
if args.drop_gnn:
    model = DropMPNN().to(device)
else:
    model = MPNN().to(device)
    use_aux_loss = False
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)
mean, std = mean[0, target].to(device), std[0, target].to(device)

def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, aux_pred = model(data)
        loss = F.mse_loss(pred, data.y)
        if use_aux_loss:
            aux_loss = F.mse_loss(aux_pred.view(-1), data.y.unsqueeze(0).expand(aux_pred.size(0),-1).clone().view(-1))
            loss = 0.75*loss + 0.25*aux_loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred, aux_pred = model(data)
        error += (pred * std - data.y * std).abs().sum().item() # MAE
    return error / len(loader.dataset)

print(model.__class__.__name__)
best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error), flush=True)
