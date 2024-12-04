import mindspore
from mindspore import nn, ops 
from mindspore.common.initializer import XavierUniform

class Embed(nn.Cell):
    def __init__(self, embed_size, clu_node_dim, clu_edge_dim, trk_node_dim, trk_edge_dim):
        super(Embed, self).__init__()
        self.clu_node_embed = nn.Dense(clu_node_dim, embed_size)
        self.clu_edge_embed = nn.Dense(clu_edge_dim, embed_size)
        self.trk_node_embed = nn.Dense(trk_node_dim, embed_size)
        self.trk_edge_embed = nn.Dense(trk_edge_dim, embed_size)

    def construct(self, clu_ndata, clu_edata, trk_ndata, trk_edata):
        h_clu = self.clu_node_embed(clu_ndata)
        e_clu = self.clu_edge_embed(clu_edata)
        h_trk = self.trk_node_embed(trk_ndata)
        e_trk = self.trk_edge_embed(trk_edata)
        return h_clu, e_clu, h_trk, e_trk

class LLPB(nn.Cell):
    def __init__(self, embed_size):
        super(LLPB, self).__init__()
        self.phi_e = nn.SequentialCell([
            nn.Dense(3*embed_size, 3*embed_size, bias_init=0),
            nn.BatchNorm1d(3*embed_size),
            nn.ReLU(),
            nn.Dense(3*embed_size, embed_size, bias_init=0),
            nn.ReLU()
        ])

        layer = nn.Dense(embed_size, 1, has_bias=False, weight_init=XavierUniform(gain=0.001))
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
        print(h[i].shape, h[j].shape, e.shape)
        out = ops.Concat(1)((h[i], h[j], e))
        print(out.shape)
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

class HeteroNet(nn.Cell):
    def __init__(self, embed_size, clu_node_dim, clu_edge_dim, trk_node_dim, trk_edge_dim,
                 n_class = 5, n_layers = 6, c_weight = 1e-3, dropout = 0.1):
        super(HeteroNet, self).__init__()
        self.embed_size = embed_size
        self.n_layers   = n_layers
        self.embed = Embed(embed_size, clu_node_dim, clu_edge_dim, trk_node_dim, trk_edge_dim)

        self.LLPB_clu = nn.SequentialCell([LLPB(embed_size) for i in range(n_layers)])
        self.LLPB_trk = nn.SequentialCell([LLPB(embed_size) for i in range(n_layers)])

        self.combine_h  = nn.SequentialCell(  nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU(),
                                        nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU())

        self.combine_e  = nn.SequentialCell(  nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU(),
                                        nn.Dense(embed_size, embed_size, bias_init=0),
                                        nn.ReLU())

        self.graph_dec = nn.SequentialCell([
            nn.Dense(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(p=1-dropout),
            nn.Dense(embed_size, n_class)
        ])

    def construct(self, clu_ndata, clu_edata, clu_src, clu_dst, trk_ndata, trk_edata, trk_src, trk_dst):
        h_clu, e_clu, h_trk, e_trk = self.embed(clu_ndata, clu_edata, trk_ndata, trk_edata)
        print(h_clu.shape, e_clu.shape, h_trk.shape, e_trk.shape)
        clu_src = clu_src.astype(mindspore.int32)
        clu_dst = clu_dst.astype(mindspore.int32)
        trk_src = trk_src.astype(mindspore.int32)
        trk_dst = trk_dst.astype(mindspore.int32)
        for i in range(self.n_layers):
            h_clu, e_clu = self.LLPB_clu[i](clu_src, clu_dst, h_clu, e_clu)
            print(h_clu.shape, e_clu.shape)
            h_trk, e_trk = self.LLPB_trk[i](trk_src, trk_dst, h_trk, e_trk)
            print(h_trk.shape, e_trk.shape)
            h = ops.Concat(0)((h_clu, h_trk))
            e = ops.Concat(0)((e_clu, e_trk))
            h = self.combine_h(h)
            e = self.combine_e(e)
            h_clu = h[:len(clu_ndata)]
            h_trk = h[len(trk_ndata):]
            e_clu = e[:len(clu_edata)]
            e_trk = e[len(trk_edata):]
        h = ops.Reshape()(h, (-1, len(clu_ndata)+len(trk_ndata), self.embed_size))
        h = ops.ReduceMean(keep_dims=False)(h, 1)
        pred = self.graph_dec(h)
        return pred


