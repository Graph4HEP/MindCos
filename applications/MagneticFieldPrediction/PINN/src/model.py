import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor
import mindspore.numpy as mnp

def sine_activation(x):
    return ms.ops.Sin()(x)

class PINN(nn.Cell):
    def __init__(self, units, activation=sine_activation):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Dense(3, units)
        self.hidden_layer2 = nn.Dense(units, units)
        self.hidden_layer3 = nn.Dense(units, units)
        self.hidden_layer4 = nn.Dense(units, units)
        self.hidden_layer5 = nn.Dense(units, 3)
        self.activation = activation

    def construct(self, inputs):
        x = inputs
        h1 = self.hidden_layer1(x)
        h1 = self.activation(h1)
        h2 = self.hidden_layer2(h1)
        h2 = self.activation(h2)
        h3 = self.hidden_layer3(h2 + h1)
        h3 = self.activation(h3)
        h4 = self.hidden_layer4(h3 + h2 + h1)
        h4 = self.activation(h4)
        h5 = self.hidden_layer5(h4 + h3 + h2 + h1)
        return h5

class PINN_Loss(nn.Cell):
    def __init__(self, N_f, L, device, addBC):
        super(PINN_Loss, self).__init__()
        self.N_f = N_f
        self.L = L
        self.device = device
        self.addBC = addBC
        self.grad = ops.GradOperation(get_all=True, get_by_list=True)
        self.sine = ms.ops.Sin()

    def construct(self, data, pred, labels, model):
        train_x = data[:,0].view(-1,1)
        train_y = data[:,1].view(-1,1)
        train_z = data[:,2].view(-1,1)
        B = model(ms.concat((train_x, train_y, train_z), axis=1))
        B_x = B[:,0]
        B_y = B[:,1]
        B_z = B[:,2]
        dx = self.grad(B_x, train_x)[0]
        dy = self.grad(B_y, train_y)[0]
        dz = self.grad(B_z, train_z)[0]
        dzy = self.grad(B_z, train_y)[0]
        dzx = self.grad(B_z, train_x)[0]
        dyz = self.grad(B_y, train_z)[0]
        dyx = self.grad(B_y, train_x)[0]
        dxy = self.grad(B_x, train_y)[0]
        dxz = self.grad(B_x, train_z)[0]

        loss_BC_div = ms.ops.ReduceMean()(ms.ops.Pow()(dx + dy + dz, 2))
        loss_BC_cul = ms.ops.ReduceMean()(ms.ops.Pow()((dzy - dyz) + (dxz - dzx) + (dyx - dxy), 2))

        x_f = mnp.random.uniform(-L/2, L/2, (N_f, 1)).astype(ms.float32)
        y_f = mnp.random.uniform(-L/2, L/2, (N_f, 1)).astype(ms.float32)
        z_f = mnp.random.uniform(-L/2, L/2, (N_f, 1)).astype(ms.float32)
        temp_pred = model(ms.concat((Tensor(x_f), Tensor(y_f), Tensor(z_f)), axis=1))
        temp_ux = temp_pred[:,0]
        temp_uy = temp_pred[:,1]
        temp_uz = temp_pred[:,2]
        u_x = self.grad(temp_ux, x_f)[0]
        u_y = self.grad(temp_uy, y_f)[0]
        u_z = self.grad(temp_uz, z_f)[0]
        u_zy = self.grad(temp_uz, y_f)[0]
        u_zx = self.grad(temp_uz, x_f)[0]
        u_yz = self.grad(temp_uy, z_f)[0]
        u_yx = self.grad(temp_uy, x_f)[0]
        u_xz = self.grad(temp_ux, z_f)[0]
        u_xy = self.grad(temp_ux, y_f)[0]

        f = ms.ops.ReduceMean()(ms.ops.Pow()(u_x + u_y + u_z, 2))
        loss_f = ms.ops.ReduceMean()(ms.ops.Pow()(f, 2))
        loss_cross = ms.ops.ReduceMean()(ms.ops.Pow()((u_zy - u_yz) + (u_xz - u_zx) + (u_yx - u_xy), 2))
        loss_u = ms.ops.ReduceMean()(ms.ops.Pow()(pred - labels[:,:,0], 2))

        if self.addBC:
            loss = loss_f + loss_u + loss_cross + loss_BC_div + loss_BC_cul
        else:
            loss = loss_f + loss_u + loss_cross
        return loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss
