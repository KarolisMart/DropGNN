
# Code to import GIN PTC dataset version https://github.com/weihua916/powerful-gnns/blob/master/util.py
import torch
import os
import shutil
import numpy as np
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def S2V_to_PyG(data):
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())

    return new_data


def load_data(dataset, degree_as_tag, folder):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (folder, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return [S2V_to_PyG(datum) for datum in g_list]


class PTCDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            name,
            transform=None,
            pre_transform=None,
    ):
        self.name = name
        self.url = 'https://github.com/weihua916/powerful-gnns/raw/master/dataset.zip'

        super(PTCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def num_tasks(self):
        return 1

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    @property
    def raw_file_names(self):
        return ['PTC.mat', 'PTC.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        folder = os.path.join(self.root, self.name)
        path = download_url(self.url, folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)

        shutil.move(os.path.join(folder, f'dataset/{self.name}'), os.path.join(folder, self.name))
        shutil.rmtree(os.path.join(folder, 'dataset'))

        os.rename(os.path.join(folder, self.name), self.raw_dir)

    def process(self):
        data_list = load_data('PTC', degree_as_tag=False, folder=self.raw_dir)
        print(sum([data.num_nodes for data in data_list]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
