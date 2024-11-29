import logging

#mindspore packages
from mindspore import Tensor, ops
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform

class LGEB(nn.Cell):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # dims for Minkowski norm & inner product

        # Define the edge feature transformation network (phi_e)
        self.phi_e = nn.SequentialCell([
            nn.Dense(n_input * 2 + n_edge_attr, n_hidden, has_bias=False),
            nn.BatchNorm1d(n_hidden),
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
