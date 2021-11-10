# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
# Datasets are implemented based on the description in the corresonding papers (see the paper for references)
import argparse
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
torch.set_printoptions(profile="full")


# Synthetic datasets

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass

class LimitsOne(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16 # There are two connected components, each with 8 nodes
        
        ports = [1,1,2,2] * 8
        colors = [0, 1, 2, 3] * 4

        y = torch.tensor([0]* 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2, 2,3,3,0, 4,5,5,6, 6,7,7,4, 8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,8], [1,0,2,1, 3,2,0,3, 5,4,6,5, 7,6,4,7, 9,8,10,9,11,10,12,11,13,12,14,13,15,14,8,15]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        return [data]

class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16 # There are two connected components, each with 8 nodes

        ports = ([1,1,2,2,1,1,2,2] * 2 + [3,3,3,3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 1,3,5,7, 8,9,9,10,10,11,11,8, 12,13,13,14,14,15,15,12, 9,15,11,13], [1,0,2,1,3,2,0,3, 5,4,6,5,7,6,4,7, 3,1,7,5, 9,8,10,9,11,10,8,11, 13,12,14,13,15,14,12,15, 15,9,13,11]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        return [data]

class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False

    def makedata(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0]==n]:
                    for nb2 in data.edge_index[1][data.edge_index[0]==n]:
                        if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = torch.tensor(labels)

        data = self.addports(data)
        data = self.makefeatures(data)
        return [data]

class LCC(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False

    def makedata(self):
        generated = False
        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)
                if nx.is_connected(nx_g):
                    i += 1
                    data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [int(nb) for nb in data.edge_index[1][data.edge_index[0]==n]]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                                    edges += 1
                        lbls[n] = int(edges/2)
                    data.y = torch.tensor(lbls)
                    labels.extend(lbls)
                    data = self.addports(data)
                    data = self.makefeatures(data)
                    graphs.append(data)
            generated = labels.count(0) >= 10 and labels.count(1) >= 10 and labels.count(2) >= 10 # Ensure the dataset is somewhat balanced

        return graphs

class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(np.logical_and(top, bottom))

    def makedata(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses

class SkipCircles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10 # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True
        self.makedata()

    def makedata(self):
        size=self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        for s, skip in enumerate(skips):
            edge_index = torch.tensor([[0, size-1], [size-1, 0]], dtype=torch.long)
            for i in range(size - 1):
                e = torch.tensor([[i, i+1], [i+1, i]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            for i in range(size):
                e = torch.tensor([[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = torch.tensor(s)
            graphs.append(data)

        return graphs        

def main(args, cluster=None):
    print(args, flush=True)

    if args.dataset == "skipcircles":
        dataset = SkipCircles()
    elif args.dataset == "triangles":
        dataset = Triangles()
    elif args.dataset == "lcc":
        dataset = LCC()
    elif args.dataset == "limitsone":
        dataset = LimitsOne()
    elif args.dataset == "limitstwo":
        dataset = LimitsTwo()
    elif args.dataset == "fourcycles":
        dataset = FourCycles()

    print(dataset.__class__.__name__)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = dataset.num_nodes
    print(f'Number of nodes: {n}')
    gamma = n
    p_opt = 2 * 1 /(1+gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    degs = []
    for g in dataset.makedata():
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    print(f'Number of graphs: {len(dataset.makedata())}')

    graph_classification = dataset.graph_class
    if graph_classification:
        print('Graph Clasification Task')
    else:
        print('Node Clasification Task')
    
    num_features = dataset.num_features
    Conv = GINConv
    if args.augmentation == 'ports':
        Conv = GINEConv
    elif args.augmentation == 'ids':
        num_features += 1
    elif args.augmentation == 'random':
        num_features += 1

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            dim = dataset.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(Conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(Conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            if args.augmentation == 'ids':
                x = torch.cat([x, data.id.float()], dim=1)
            elif args.augmentation == 'random':
                x = torch.cat([x, torch.randint(0, 100, (x.size(0), 1), device=x.device) / 100.0], dim=1)
            
            outs = [x]
            for i in range(self.num_layers):
                if args.augmentation == 'ports':
                    x = self.convs[i](x, edge_index, data.ports.expand(-1, x.size(-1)))
                else:
                    x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)
            
            out = None
            for i, x in enumerate(outs):
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x) # No dropout for these experiments
                if out is None:
                    out = x
                else:
                    out += x
            return F.log_softmax(out, dim=-1), 0

    use_aux_loss = args.use_aux_loss

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            dim = dataset.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(Conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(Conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
            
            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, dataset.num_classes))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            
            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
            x[drop] = 0.0
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):   
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index

            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x) # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x
            
            if use_aux_loss:
                aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    if graph_classification:
                        x = x.view(-1, x.size(-1))
                        x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = self.aux_fcs[i](x) # No dropout layer in these experiments
                    aux_out += x

                return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return F.log_softmax(out, dim=-1), 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.augmentation == 'dropout':
        model = DropGIN().to(device)
    else:
        model = GIN().to(device)
        use_aux_loss = False

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        n = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logs, aux_logs = model(data)
            loss = F.nll_loss(logs, data.y)
            n += len(data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(0).expand(aux_logs.size(0),-1).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        n = 0
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data)
                pred = logs.max(1)[1]
                n += len(pred)
                correct += pred.eq(data.y).sum().item()
        return correct / n

    def train_and_test(multiple_tests=False, test_over_runs=None):
        train_accs = []
        test_accs = []
        nonlocal num_runs # access global num_runs variable inside this function
        print(model.__class__.__name__)
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset_parameters()
            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            test_dataset = dataset.makedata()
            train_dataset = dataset.makedata()

            test_loader = DataLoader(test_dataset, batch_size=len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

            print('---------------- Seed {} ----------------'.format(seed))
            for epoch in range(1, 1001):
                if args.verbose:
                    start = time.time()
                train_loss = train(epoch, train_loader, optimizer)
                if args.verbose:
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Time: {:7f}'.format(epoch, lr, train_loss, time.time() - start), flush=True)
            train_acc = test(train_loader)
            train_accs.append(train_acc)
            if not test_over_runs is None:
                if multiple_tests:
                    for i in range(10):
                        old_num_runs = num_runs
                        for r in test_over_runs:
                            num_runs = r
                            test_acc = test(test_loader)
                            test_accs.append(test_acc)
                        num_runs = old_num_runs
                else:
                    old_num_runs = num_runs
                    for r in test_over_runs:
                        num_runs = r
                        test_acc = test(test_loader)
                        test_accs.append(test_acc)
                    num_runs = old_num_runs
            elif multiple_tests:
                for i in range(10):
                    test_acc = test(test_loader)
                    test_accs.append(test_acc)
                test_acc =  torch.tensor(test_accs[-10:]).mean().item()
            else:
                test_acc = test(test_loader)
                test_accs.append(test_acc)
            print('Train Acc: {:.7f}, Test Acc: {:7f}'.format(train_acc, test_acc), flush=True)            
        train_acc = torch.tensor(train_accs)
        test_acc = torch.tensor(test_accs)
        if not test_over_runs is None:
            test_acc = test_acc.view(-1, len(test_over_runs))
        print('---------------- Final Result ----------------')
        print('Train Mean: {:7f}, Train Std: {:7f}, Test Mean: {}, Test Std: {}'.format(train_acc.mean(), train_acc.std(), test_acc.mean(dim=0), test_acc.std(dim=0)), flush=True)
        return test_acc.mean(dim=0), test_acc.std(dim=0)

    if args.prob_ablation:
        print('Dropout probability ablation')
        probs = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.95]
        means = []
        stds = []
        for prob in probs:
            print(f'Dropout probability {prob}:')
            p = prob
            mean, std = train_and_test(multiple_tests=True)
            means.append(mean.item())
            stds.append(std.item())
        probs = np.array(probs)
        means = np.array(means)
        stds = np.array(stds)
        lower = means - stds
        lower = [i if i > 0 else 0 for i in lower]
        upper = means + stds
        upper = [i if i <= 1 else 1 for i in upper]
        plt.plot(probs, means)
        plt.fill_between(probs, lower, upper, alpha=0.3)
        plt.xlabel("Dropout Probability")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.ylim(bottom=0.4)
        plt.vlines(p_opt, 0, 2, colors="k")
        file_name = "ablation_p_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    elif args.num_runs_ablation:
        print('Run count ablation')
        runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 35, 50]
        means, stds = train_and_test(multiple_tests=True, test_over_runs=runs)
        runs = np.array(runs)
        means = means.numpy()
        stds = stds.numpy()
        lower = means - stds
        lower = [i if i > 0 else 0 for i in lower]
        upper = means + stds
        upper = [i if i <= 1 else 1 for i in upper]
        plt.plot(runs, means)
        plt.fill_between(runs, lower, upper, alpha=0.3)
        plt.tight_layout()
        plt.xlabel("Number of Runs")
        plt.ylabel("Accuracy")
        plt.ylim(bottom=0.4)
        file_name = "ablation_runs_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    else:
        train_and_test()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', type=str, default='none', help="Options are ['none', 'ports', 'ids', 'random', 'dropout']")
    parser.add_argument('--prob', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=4) # 9 layers were used for skipcircles dataset
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--prob_ablation', action='store_true', default=False, help="Run probability ablation study")
    parser.add_argument('--num_runs_ablation', action='store_true', default=False, help="Run number of runs ablation study")

    parser.add_argument('--dataset', type=str, default='limitsone', help="Options are ['skipcircles', 'triangles', 'lcc', 'limitsone', 'limitstwo', 'fourcycles']")
    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)
