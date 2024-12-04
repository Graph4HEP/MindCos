import dgl
import mindspore.dataset as mds


class LLPDataset(mds.Dataset):
    def __init__(self, fname):
        self.graphs, labels = dgl.load_graphs(fname)
        self.labels = labels['labels'].numpy()
    def __getitem__(self, idx):
        return (self.graphs[idx], self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def build(self):
        ds = mds.NumpySlicesDataset((self.label, self.graphs), column_names=['label', 'p4s', 'nodes', 'atom_mask'], sampler=mds.RandomSampler())
