import numpy as np
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore import Tensor
from mindspore import Parameter
from mindspore import ops
from mindspore import Parameter
from mindspore.common import dtype as mstype


class Embed(nn.Cell):  
    def __init__(self, embed_size):  
        super(Embed, self).__init__()  
        self.clu_node_embed = nn.Dense(6, embed_size)  
        self.clu_edge_embed = nn.Dense(8, embed_size)  
  
    def construct(self, ndata, edata):  
        h_clu = self.clu_node_embed(ndata)  
        e_clu = self.clu_edge_embed(edata)  
        return h_clu, e_clu  

class NN(nn.Cell):  
    def __init__(self, embed_size):  
        super(NN, self).__init__()  
        self.phi_e = nn.SequentialCell([  
            nn.Dense(3*embed_size, 3*embed_size, bias_init=0),  
            nn.BatchNorm1d(3*embed_size),  
            nn.ReLU(),  
            nn.Dense(3*embed_size, embed_size, bias_init=0),  
            nn.ReLU()  
        ])  

        layer = nn.Dense(embed_size, 1, bias_init=0)
        layer_weight = Tensor(np.random.normal(scale=0.01, size=(embed_size, 1)).astype(np.float32))
        layer_weight = layer_weight.astype(mindspore.common.dtype.float32)
        layer.weight = Parameter(layer_weight)  # 省略name参数或传递None
        self.phi_x = nn.SequentialCell([  
            nn.Dense(embed_size, embed_size, bias_init=0),  
            nn.BatchNorm1d(embed_size),  
            nn.ReLU(),  
            layer
        ])          
        self.phi_m = nn.SequentialCell([  
            nn.Dense(embed_size, 1, bias_init=0),  
            nn.Sigmoid()  
        ])  
        self.phi_h = nn.SequentialCell([  
            nn.Dense(2*embed_size, 2*embed_size, bias_init=0),  
            nn.BatchNorm1d(2*embed_size),  
            nn.ReLU(),  
            nn.Dense(2*embed_size, embed_size, bias_init=0)  
        ])  
    # m = phi_e(hi,hj,e)  
    def m_model(self, i, j, h, e):  
        out = ops.Concat(1)((h[i], h[j], e))  
        out = self.phi_e(out)  
        return out  
    # xij = xij + phi_m(mij)xij  
    def x_model(self, i, j, e, m):  
        trans = e * self.phi_m(m)  
        trans = ops.clip_by_value(trans, -100,100)
        e = e + trans   
        return e  
    def h_model(self, i, j, h, m):  
        phim = m * self.phi_m(m)  
        agg = ops.UnsortedSegmentSum()(phim, i, h.shape[0])
        agg = ops.Concat(1)((h, agg))  
        out = h + self.phi_h(agg)  
        return out  
    def construct(self, i, j, h, e):  
        m = self.m_model(i, j, h, e)  
        e = self.x_model(i, j, e, m)  
        h = self.h_model(i, j, h, m)  
        return h, e

class MLPpred(nn.Cell):
    def __init__(self, h_feats, out_class):
        super().__init__()
        self.W1 = nn.Dense(h_feats * 2, h_feats)
        self.W2 = nn.Dense(h_feats, out_class)

    def apply_edges(self, src_data, dst_data):
        h = ops.Concat(1)((src_data, dst_data))
        return self.W2(ops.ReLU()(self.W1(h))).squeeze(1)

    def construct(self, src, dst, h):
        src_data = h[src]
        dst_data = h[dst]
        edge_feat = self.apply_edges(src_data, dst_data)
        return edge_feat

class Finding(nn.Cell):
    def __init__(self, embed_size):
        super(Finding, self).__init__()
        self.embed_size = embed_size
        self.embed = Embed(embed_size)

        self.LLPB1 = NN(embed_size)
        self.LLPB2 = NN(embed_size)
        self.LLPB3 = NN(embed_size)

        self.combine_h  = nn.SequentialCell(  nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU(),
                                        nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU())

        self.combine_e  = nn.SequentialCell(  nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU(),
                                        nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU())

        self.graph_dec = MLPpred(embed_size, 1)

    def construct(self, ndata, edata, src, dst):
        h_clu, e_clu= self.embed(ndata, edata)
        src = src.astype(mindspore.int32)
        dst = dst.astype(mindspore.int32)  
        h_clu, e_clu = self.LLPB1(src, dst, h_clu, e_clu)
        h_clu = self.combine_h(h_clu)
        e_clu = self.combine_e(e_clu)
        h_clu, e_clu = self.LLPB2(src, dst, h_clu, e_clu)
        h_clu = self.combine_h(h_clu)
        e_clu = self.combine_e(e_clu)
        h_clu, e_clu = self.LLPB3(src, dst, h_clu, e_clu)
        h_clu = self.combine_h(h_clu)
        e_clu = self.combine_e(e_clu)
        pred = self.graph_dec(src, dst, h_clu)
        return pred

if __name__ == '__main__':
    net = Finding(128)
    print('model can be built correctly') 
    tensor_n = np.random.rand(2,6)
    print(tensor_n)
    tensor_e = np.random.rand(1,8)
    print(tensor_e)
    src = np.array([0,1])
    dst = np.array([1,0])
    out = net(tensor_n, tensor_e, src, dst)
    print(out)