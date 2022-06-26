# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import StratifiedKFold
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
from ptc_dataset import PTCDataset


def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size

    if 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70
        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=69).to(torch.float)#136 in k-gnn?
                return data
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
        dataset = TUDataset(
            path,
            name=args.dataset,
            pre_transform=MyPreTransform(),
            pre_filter=MyFilter())
    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MUTAG')
        dataset = TUDataset(path, name='MUTAG', pre_filter=MyFilter())
    elif 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
        dataset = TUDataset(path, name='PROTEINS', pre_filter=MyFilter())
    elif 'PTC_GIN' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True        
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PTC_GIN')
        dataset = PTCDataset(path, name='PTC', pre_filter=MyFilter())
    else:
        raise ValueError

    print(dataset)

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

    def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            outs = [x]
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)
            
            out = None
            for i, x in enumerate(outs):
                x = global_add_pool(x, batch)
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                if out is None:
                    out = x
                else:
                    out += x
            return F.log_softmax(out, dim=-1), 0

    use_aux_loss = args.use_aux_loss

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
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
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            
            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del  run_edge_index
            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                x = global_add_pool(x, batch)
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                if out is None:
                    out = x
                else:
                    out += x

            if use_aux_loss:
                aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    x = x.view(-1, x.size(-1))
                    x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = F.dropout(self.aux_fcs[i](x), p=self.dropout, training=self.training)
                    aux_out += x

                return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return F.log_softmax(out, dim=-1), 0

    torch.manual_seed(0)
    np.random.seed(0)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.drop_gnn:
        model = DropGIN().to(device)
    else:
        model = GIN().to(device)
        use_aux_loss = False

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logs, aux_logs = model(data)
            loss = F.nll_loss(logs, data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(0).expand(aux_logs.size(0),-1).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len(loader.dataset)

    def val(loader):
        model.eval()
        with torch.no_grad():
            loss_all = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data)
                loss_all += F.nll_loss(logs, data.y, reduction='sum').item()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data)
                pred = logs.max(1)[1]
                correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    acc = []
    splits = separate_data(len(dataset), seed=0)
    print(model.__class__.__name__)
    for i, (train_idx, test_idx) in enumerate(splits):
        model.reset_parameters()
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # in GIN code 50 itters per epoch were used

        test_dataset = dataset[test_idx.tolist()]
        train_dataset = dataset[train_idx.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=BATCH)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*50/(len(train_dataset)/BATCH))), batch_size=BATCH, drop_last=False, collate_fn=Collater(follow_batch=[],exclude_keys=[]))	# GIN like epochs/batches - they do 50 radom batches per epoch

        print('---------------- Split {} ----------------'.format(i), flush=True)

        test_acc = 0
        acc_temp = []
        for epoch in range(1, 350+1):
            if args.verbose or epoch == 350:
                start = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(epoch, train_loader, optimizer)
            scheduler.step()
            test_acc = test(test_loader)
            if args.verbose or epoch == 350:
                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
                        epoch, lr, train_loss, 0, test_acc, time.time() - start), flush=True)
            acc_temp.append(test_acc)
        acc.append(torch.tensor(acc_temp))
    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print('---------------- Final Epoch Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,-1].mean(), acc[:,-1].std()))
    print(f'---------------- Best Epoch: {best_epoch} ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,best_epoch].mean(), acc[:,best_epoch].std()), flush=True)

if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.5, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 128])
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32]) # 64 is used for social datasets (IMDB) and 16 or 32 for bio datasest (MUTAG, PTC, PROTEINS). Set tunable=False to not grid search over this (for social datasets)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--drop_gnn', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG', help="Options are ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")

    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '24:00:00'
        
        cluster.memory_mb_per_node = '12G'
        job_name = args.dataset+f'{"_DropGNN" if args.drop_gnn else ""}'+f'{"_aux_loss" if args.drop_gnn and args.use_aux_loss else ""}'
        if args.gpu_jobs:
            cluster.per_experiment_nb_cpus = 2
            cluster.per_experiment_nb_gpus = 1
            cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
        else:
            cluster.per_experiment_nb_cpus = 8
            cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)
