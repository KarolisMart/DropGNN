# This implementation is based on https://github.com/chrsmrrs/k-gnn/blob/master/examples/1-qm9.py
import os.path as osp
import argparse
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree


parser = argparse.ArgumentParser()
parser.add_argument('--target', default=0)
parser.add_argument('--drop_gnn', action='store_true', default=False)
parser.add_argument('--aux_loss', action='store_true', default=False)
args = parser.parse_args()
print(args)
target = int(args.target)
print('---- Target: {} ----'.format(target))

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu
        return data

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', '1-QM9')
dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))

dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
test_loader = DataLoader(test_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Set the sampling probability and number of runs/samples for the DropGIN
n = []
degs = []
for g in dataset:
    num_nodes = g.num_nodes
    deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
    n.append(g.num_nodes)
    degs.append(deg.max())
print(f'Mean Degree: {torch.stack(degs).mean()}')
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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        M_in, M_out = dataset.num_features, 32
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))

        x = scatter_mean(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1), None

class DropNet(torch.nn.Module):
    def __init__(self):
        super(DropNet, self).__init__()
        M_in, M_out = dataset.num_features, 32
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

        if args.aux_loss:
            self.fc1_aux = torch.nn.Linear(64, 32)
            self.fc2_aux = torch.nn.Linear(32, 16)
            self.fc3_aux = torch.nn.Linear(16, 1)

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
        x = F.elu(self.conv1(x, run_edge_index, run_edge_attr))
        x = F.elu(self.conv2(x, run_edge_index, run_edge_attr))
        x = F.elu(self.conv3(x, run_edge_index, run_edge_attr))
        del run_edge_index, run_edge_attr
        if args.aux_loss:
            run_batch = data.batch.repeat(num_runs) + torch.arange(num_runs, device=data.batch.device).repeat_interleave(data.batch.size(0)) * (data.batch.max() + 1)
            aux_x = scatter_mean(x, run_batch, dim=0)
            del run_batch
            aux_x = F.elu(self.fc1_aux(aux_x))
            aux_x = F.elu(self.fc2_aux(aux_x))
            aux_x = self.fc3_aux(aux_x)
            aux_x = aux_x.view(num_runs, -1, aux_x.size(-1))
        x = x.view(num_runs, -1, x.size(-1))
        x = x.mean(dim=0)
        
        x = scatter_mean(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        
        return x.view(-1), aux_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_aux_loss = args.aux_loss
if args.drop_gnn:
    model = DropNet().to(device)
else:
    model = Net().to(device)
    use_aux_loss = False
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=5, min_lr=0.00001)

mean, std = mean[target].to(device), std[target].to(device)

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
        loss_all += loss .item()* data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred, aux_pred = model(data)
        error += ((pred * std) -
                  (data.y * std)).abs().sum().item() # MAE
    return error / len(loader.dataset)

print(model.__class__.__name__)
best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None:
        best_val_error = val_error
    if val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error
    
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error), flush=True)
