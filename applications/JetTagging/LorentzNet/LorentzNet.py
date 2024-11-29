#necessary packages
import numpy as np
import energyflow
from sklearn.preprocessing import OneHotEncoder
import time
import os,sys
import warnings
GPU_ID = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
warnings.filterwarnings('ignore')
import logging

#mindspore packages
import mindspore.dataset as mds
import mindspore.dataset.transforms.c_transforms as C
import mindspore as ms
from mindspore import Tensor, ops
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import XavierUniform
from mindspore import context

device = sys.argv[2]
context.set_context(mode=1, device_target=device)

# 定义 MindSpore 的数据集类
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
        # 获取单个样本
        return (self.label[idx], self.p4s[idx], self.nodes[idx], self.atom_mask[idx])

    def __len__(self):
        # 数据集大小
        return len(self.label)

    def build(self, column_names=None):
        # 构建数据集
        ds = mds.NumpySlicesDataset((self.label, self.p4s, self.nodes, self.atom_mask), column_names=['label', 'p4s', 'nodes', 'atom_mask'], sampler=mds.RandomSampler())
        #ds = mds.NumpySlicesDataset((self.label, self.p4s, self.nodes, self.atom_mask), column_names=['label', 'p4s', 'nodes', 'atom_mask'], sampler=None, shuffle=False)
        # 设置 batch 大小和重复次数
        ds = ds.batch(self.batch_size, drop_remainder=True).repeat(self.repeat_size)
        return ds

    def map(self):
        # 设置并行工作数
        ds = self.build()
        ds = ds.map(operations=self.collate_fn, input_columns=['label', 'p4s', 'nodes', 'atom_mask'], output_columns=['label', 'p4s', 'nodes', 'atom_mask', 'edge_mask', 'edges'],
                    num_parallel_workers=self.num_parallel_workers)
        return ds

    @staticmethod
    def collate_fn(label, p4s, nodes, atom_mask):
        # 定义 collate_fn 函数，与 PyTorch中的 collate_fn 类似
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
        # 定义 get_adj_matrix 函数，与 PyTorch 中的 get_adj_matrix 类似
        rows, cols = [], []
        for batch_idx in range(batch_size):
            nn = batch_idx * n_nodes
            x = edge_mask[batch_idx]
            rows.append(nn + np.where(x)[0])
            cols.append(nn + np.where(x)[1])
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

class LGEB(nn.Cell):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # dims for Minkowski norm & inner product

        # Define the edge feature transformation network (phi_e)
        self.phi_e = nn.SequentialCell([
            nn.Dense(n_input * 2 + n_edge_attr, n_hidden, has_bias=False),
            #nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dense(n_hidden, n_hidden),
            nn.ReLU()
        ])

        # Define the hidden state transformation network (phi_h)
        self.phi_h = nn.SequentialCell([
            nn.Dense(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dense(n_hidden, n_output)
        ])

        # Define the transformation network for x (phi_x)
        layer = nn.Dense(n_hidden, 1, has_bias=False, weight_init=XavierUniform(gain=0.001))
        self.phi_x = nn.SequentialCell([
            nn.Dense(n_hidden, n_hidden),
            nn.ReLU(),
            layer
        ])

        # Define the transformation network for m (phi_m)
        self.phi_m = nn.SequentialCell([
            nn.Dense(n_hidden, 1),
            nn.Sigmoid()
        ])

        self.last_layer = last_layer
        if last_layer:
            self.phi_x = None

    def m_model(self, hi, hj, norms, dots):
        out = ops.Concat(axis=1)([hi, hj, norms, dots])
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        agg = ops.unsorted_segment_sum(m, i, num_segments=h.shape[0])
        agg = ops.Concat(axis=1)([h, agg, node_attr])
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        trans = ops.clamp(trans, min=-100, max=100)
        agg = ops.unsorted_segment_sum(trans, i, num_segments=x.shape[0])
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = ops.Sub()(x[i], x[j])
        norms = self.normsq4(x_diff).view((-1, 1))
        dots = self.dotsq4(x[i], x[j]).view((-1, 1))
        norms, dots = self.psi(norms), self.psi(dots)
        return norms, dots, x_diff

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result = Tensor([0])
        result = result.new_zeros((num_segments, data.shape[1]))
        result.index_add_(result, segment_ids, data)
        return result

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result = Tensor([0])
        result = result.new_zeros((num_segments, data.shape[1]))
        count = Tensor([0])
        count = count.new_zeros((num_segments, data.shape[1]))
        result.index_add_(result, segment_ids, data)
        count.index_add_(count, segment_ids, Tensor.ones_like(data))
        return result / ops.Minimum()(count, Tensor.ones_like(count))

    def normsq4(self, p):
        psq = ops.Pow()(p, 2)
        return 2 * psq[..., 0] - ops.ReduceSum()(psq, -1)

    def dotsq4(self, p, q):
        psq = ops.Mul()(p, q)
        return 2 * psq[..., 0] - ops.ReduceSum()(psq, -1)

    def psi(self, p):
        return ops.Sign()(p) * ops.Log()(ops.Abs()(p) + 1)

    def construct(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        #print('inut:',norms.mean(), dots.mean(), x_diff.mean())
        logging.info('inut:',norms.mean(), dots.mean(), x_diff.mean())
        #print('h:', h.mean())
        logging.info('h:', h.mean())
        m = self.m_model(h[i], h[j], norms, dots)  # [B*N, hidden]
        #print('m', m.mean())
        logging.info('m', m.mean())
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
            #print('x:', x.mean())
            logging.info('x:', x.mean())
        h = self.h_model(h, edges, m, node_attr)
        #print('h:',h.mean())
        logging.info('h:',h.mean())
        return h, x, m

class LorentzNet(nn.Cell):
    r''' Implementation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, n_scalar, n_hidden, n_class=2, n_layers=6, c_weight=1e-3, dropout=0.1):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Dense(n_scalar, n_hidden, has_bias=True)
        self.LGEBs = nn.CellList([LGEB(self.n_hidden, self.n_hidden, self.n_hidden,
                                    n_node_attr=n_scalar, dropout=dropout,
                                    c_weight=c_weight, last_layer=(i == n_layers - 1))
                                    for i in range(n_layers)])
        self.graph_dec = nn.SequentialCell([
            nn.Dense(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(p=1-dropout),
            nn.Dense(self.n_hidden, n_class)
        ])

    def construct(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        #print('scalars:', scalars.mean())
        logging.info('scalars:', scalars.mean())
        h = self.embedding(scalars)
        #print('h embed:',h.mean())
        logging.info('h embed:',h.mean())
        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=scalars)
            #print(h.shape, x.shape)

        h = ops.Mul()(h, node_mask)
        #print(h.shape)
        h = ops.Reshape()(h, (-1, n_nodes, self.n_hidden))
        #print(h.shape)
        h = ops.ReduceMean(keep_dims=False)(h, 1)
        #print(h.shape)
        pred = self.graph_dec(h)
        #print(pred.shape)
        return ops.squeeze(pred)

model = LorentzNet(n_scalar = 8, n_hidden = 72, n_class = 2,
                       dropout = 0.2, n_layers = 6,
                       c_weight = 1e-3)

def process_bar_train(num, total, dt, loss, acc, Type=''):
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    r = '\r{} [{}{}]{}/{} - used {:.1f}s / left {:.1f}s / loss {:.10f} / acc {:.4f} '.format(Type, '*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, loss, acc)
    sys.stdout.write(r)
    sys.stdout.flush()


lr = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0008535533905932737, 0.0005, 0.00014644660940672628, 0.001, 0.0009619397662556434, 0.0008535533905932737, 0.000691341716182545, 0.0005, 0.0003086582838174551, 0.00014644660940672628, 3.806023374435663e-05, 0.001, 0.0009903926402016153, 0.0009619397662556434, 0.0009157348061512727, 0.0008535533905932737, 0.0007777851165098011, 0.000691341716182545, 0.0005975451610080642, 0.0005, 0.00040245483899193594, 0.0003086582838174551, 0.00022221488349019903, 0.00014644660940672628, 8.426519384872733e-05, 3.806023374435663e-05, 9.607359798384786e-06, 4.803679899192393e-06, 2.4018399495961964e-06, 1.2009199747980982e-06]
Nbatch = 2500
lr = [x for x in lr for _ in range(Nbatch)]
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=lr)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, label):
    logits = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_loop(model, dataloader):
    num_batches = len(dataloader)
    model.set_train()
    st = 0
    total, loss, correct = 0, 0, 0
    for i, (label, p4s, nodes, atom_mask, edge_mask, edges) in enumerate(dataloader):
        if i == 0:
            st = time.time()
        label = label.astype(ms.int32)
        p4s = p4s.astype(ms.float32)
        nodes = nodes.astype(ms.float32)
        atom_mask = atom_mask.astype(ms.float32)
        edge_mask = edge_mask.astype(ms.float32)
        edges = edges.astype(ms.int32)
        batch_size, n_nodes, _ = p4s.shape
        atom_positions = p4s.reshape(batch_size * n_nodes, -1)
        atom_mask = atom_mask.reshape(batch_size * n_nodes, -1)
        edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1)
        nodes = nodes.reshape(batch_size * n_nodes, -1)
        (losses, logits), grads = grad_fn(nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, label)
        if ops.IsNan()(grads).sum()>0:
            print(res)
            sys.exit()
        grads = ops.clip_by_norm(grads, max_norm=1)
        optimizer(grads)
        loss += losses.asnumpy()
        correct += (logits.argmax(1) == label).asnumpy().sum()
        total += len(p4s)
        #print(f"loss: {loss/(i+1):>7f} acc: {100*correct/total:>0.1f} [{i+1:>3d}/{num_batches:>3d}] time: [{time.time()-st:>0.1f}/{(time.time()-st)/(i+1)*num_batches:>0.1f}]")
        process_bar_train(i+1, num_batches, time.time()-st, loss/(i+1), 100*correct/total, '')

def test_loop(model, dataloader, loss_fn):
    num_batches =len(dataloader)
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for i, (label, p4s, nodes, atom_mask, edge_mask, edges) in enumerate(dataloader):
        label = label.astype(ms.int32)
        p4s = p4s.astype(ms.float32)
        nodes = nodes.astype(ms.float32)
        atom_mask = atom_mask.astype(ms.float32)
        edge_mask = edge_mask.astype(ms.float32)
        edges = edges.astype(ms.int32)
        batch_size, n_nodes, _ = p4s.shape
        atom_positions = p4s.reshape(batch_size * n_nodes, -1)
        atom_mask = atom_mask.reshape(batch_size * n_nodes, -1)
        edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, -1)
        nodes = nodes.reshape(batch_size * n_nodes, -1)
        pred = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        total += len(p4s)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Valid: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

dataset, dataloaders = retrieve_dataloaders(32, 100000)
print('Train')
for t in range(35):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, dataloaders['train'])
    print()
    test_loop(model, dataloaders['val'], loss_fn)

print('Test')
test_loop(model, dataloaders['test'], loss_fn)
print("Done!")
    
