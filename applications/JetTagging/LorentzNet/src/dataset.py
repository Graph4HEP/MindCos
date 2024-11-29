#necessary packages
import numpy as np
import energyflow
from sklearn.preprocessing import OneHotEncoder
import time

#mindspore packages
import mindspore.dataset as mds
import mindspore.dataset.transforms.c_transforms as C
import mindspore as ms
from mindspore import Tensor

class JetDataset(mds.Dataset):
    def __init__(self, label, p4s, nodes, atom_mask, batch_size, repeat_size=1, num_parallel_workers=1):
        self.label = label
        self.p4s = p4s
        self.nodes = nodes
        self.atom_mask = atom_mask
        self.batch_size = batch_size
        self.repeat_size = repeat_size
        self.num_parallel_workers = num_parallel_workers

    def __getitem__(self, idx):
        return (self.label[idx], self.p4s[idx], self.nodes[idx], self.atom_mask[idx])

    def __len__(self):
        return len(self.label)

    def build(self, column_names=None):
        ds = mds.NumpySlicesDataset((self.label, self.p4s, self.nodes, self.atom_mask), column_names=['label', 'p4s', 'nodes', 'atom_mask'], sampler=mds.RandomSampler())
        ds = ds.batch(self.batch_size, drop_remainder=True).repeat(self.repeat_size)
        return ds

    def map(self):
        ds = self.build()
        ds = ds.map(operations=self.collate_fn, input_columns=['label', 'p4s', 'nodes', 'atom_mask'], output_columns=['label', 'p4s', 'nodes', 'atom_mask', 'edge_mask', 'edges'],
                    num_parallel_workers=self.num_parallel_workers)
        return ds

    @staticmethod
    def collate_fn(label, p4s, nodes, atom_mask):
        batch_size = p4s.shape[0]
        n_nodes = p4s.shape[1]
        edge_mask = np.expand_dims(atom_mask, axis=1) * np.expand_dims(atom_mask, axis=2)
        diag_mask = np.eye(edge_mask.shape[1], dtype=bool)
        diag_mask = ~np.expand_dims(diag_mask, axis=0)
        edge_mask *= diag_mask
        edges = JetDataset.get_adj_matrix(n_nodes, batch_size, edge_mask)
        return label, p4s, nodes, atom_mask, edge_mask, edges

    @staticmethod
    def get_adj_matrix(n_nodes, batch_size, edge_mask):
        rows, cols = [], []
        for batch_idx in range(batch_size):
            n_n = batch_idx * n_nodes
            x = edge_mask[batch_idx]
            rows.append(n_n + np.where(x)[0])
            cols.append(n_n + np.where(x)[1])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        return rows, cols

def retrieve_dataloaders(batch_size, num_data=-1, use_one_hot=True, cache_dir='./data'):
    raw = energyflow.qg_jets.load(num_data=num_data, pad=True, ncol=4, generator='pythia',
                                  with_bc=False, cache_dir=cache_dir)
    splits = ['train', 'val', 'test']
    data = {type: {'raw': None, 'label': None} for type in splits}
    (data['train']['raw'], data['val']['raw'], data['test']['raw'],
     data['train']['label'], data['val']['label'], data['test']['label']) = \
        energyflow.utils.data_split(*raw, train=0.8, val=0.1, test=0.1, shuffle=False)

    enc = OneHotEncoder(handle_unknown='ignore').fit([[11], [13], [22], [130], [211], [321], [2112], [2212]])

    for split, value in data.items():
        pid = np.abs(np.asarray(value['raw'][..., 3], dtype=int))[..., None]
        p4s = energyflow.p4s_from_ptyphipids(value['raw'], error_on_unknown=True)
        one_hot = enc.transform(pid.reshape(-1, 1)).toarray().reshape(pid.shape[:2] + (-1,))
        one_hot = np.array(one_hot)
        mass = energyflow.ms_from_p4s(p4s)[..., None]
        charge = energyflow.pids2chrgs(pid)
        if use_one_hot:
            nodes = one_hot
        else:
            nodes = np.concatenate((mass, charge), axis=-1)
            nodes = np.sign(nodes) * np.log(np.abs(nodes) + 1)
        atom_mask = (pid[..., 0] != 0).astype(bool)
        value['p4s'] = p4s
        value['nodes'] = nodes
        value['label'] = value['label']
        value['atom_mask'] = atom_mask

    datasets = {split: JetDataset(value['label'], value['p4s'], value['nodes'], value['atom_mask'], batch_size)
                for split, value in data.items()}

    dataloaders = {split: datasets[split].map() for split, dataset in datasets.items()}

    return datasets, dataloaders

