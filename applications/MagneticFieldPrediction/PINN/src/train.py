import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.numpy as mnp
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint
import time
import matplotlib.pyplot as plt
import numpy as np
from .model import *
def lr_adjust(val_loss, optimizer):
    if val_loss < 0.002:
        optimizer.set_lr(0.0001)
    if val_loss < 0.001:
        optimizer.set_lr(0.00001)
    if val_loss < 0.0001:
        optimizer.set_lr(0.000001)

def train(train_data, train_labels, test_data, test_labels, config):
    Nep = config['Nep']
    units = config['units']
    device = config['device']
    lr = config['lr']
    L = config['length'] / 2
    path = config['path']
    Npde = config['Npde']
    adjust = config['adjust_lr']
    addBC = config['addBC']
    model = PINN(units).to(device)
    optimizer1 = ms.optim.Adam(model.trainable_params(), learning_rate=lr)
    optimizer2 = ms.optim.LBFGS(model.trainable_params(), learning_rate=lr)
    criterion = PINN_Loss(Npde, L, device, addBC)
    
    loss_f_l = []
    loss_u_l = []
    loss_cross_l = []
    loss_BC_div_l = []
    loss_BC_cul_l = []
    loss_l = []
    test_loss_l = []
    epoch = []

    mini_loss = 100000000.0
    best_model = model
    best_ep = 0

    train_data = Tensor(train_data, dtype=ms.float32).to(device)
    train_labels = Tensor(train_labels, dtype=ms.float32).to(device)
    test_data = Tensor(test_data, dtype=ms.float32).to(device)
    test_labels = Tensor(test_labels, dtype=ms.float32).to(device)
    
    st = time.time()

    for ep in range(Nep):
        model.train()
        if config['lbfgs'] == 1:
            if ep < 10000:
                optimizer = optimizer1
            else:
                optimizer = optimizer2
        else:
            optimizer = optimizer1

        optimizer.clear_grad()
        pred = model(train_data)
        loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion(train_data, pred, train_labels, model)
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            epoch.append(ep)
            loss_f_l.append(loss_f.asnumpy())
            loss_u_l.append(loss_u.asnumpy())
            loss_cross_l.append(loss_cross.asnumpy())
            loss_BC_div_l.append(loss_BC_div.asnumpy())
            loss_BC_cul_l.append(loss_BC_cul.asnumpy())
            loss_l.append(loss.asnumpy())

            model.eval()
            test_pred = model(test_data)
            test_loss = ms.ops.ReduceMean()(ms.ops.Pow()(test_pred - test_labels[:, :, 0], 2)).asnumpy()
            test_loss_l.append(test_loss)
            if adjust:
                lr_adjust(test_loss, optimizer1)
            if mini_loss > test_loss:
                best_model = model
                mini_loss = test_loss
                best_ep = ep

            if test_loss < 0.000001:
                break

        if ep % 1000 == 0:
            print(f'===>>> ep: {ep}')
            print(f'time used: {time.time() - st:.2f}s, time left: {(time.time() - st) / (ep + 1) * Nep - (time.time() - st):.2f}s')
            print(f'loss_B: {loss_u.asnumpy():.7f}, loss_div: {loss_f.asnumpy():.7f}, loss_cul: {loss_cross.asnumpy():.7f}, loss_BC_div: {loss_BC_div.asnumpy():.7f}, loss_BC_cul: {loss_BC_cul.asnumpy():.7f}')
            print(f'total loss: {loss.asnumpy():.7f}, test loss: {test_loss:.7f}')
                
    print(f'best loss at ep: {best_ep}, best_loss: {mini_loss:.7f}')
    print(f'total time used: {time.time() - st:.2f}s')
    plt.plot(epoch, loss_f_l, label='loss div')
    plt.plot(epoch, loss_u_l, label='loss B')
    plt.plot(epoch, loss_cross_l, label='loss cul')
    if config['addBC'] == 1:
        plt.plot(epoch, loss_BC_div_l, label='loss BC div')
        plt.plot(epoch, loss_BC_cul_l, label='loss BC cul')    
    plt.plot(epoch, loss_l, label='total loss')
    plt.plot(epoch, test_loss_l, label='test loss')
    plt.scatter(best_ep, mini_loss, label='test best loss', marker='*')
    plt.legend()
    plt.yscale('log')
    plt.show()
    plt.savefig(f'{path}/loss.png')
    plt.close()
    
    np.save(f'{path}/loss_div.npy', np.array(loss_f_l))
    np.save(f'{path}/loss_B.npy', np.array(loss_u_l))
    np.save(f'{path}/loss_cul.npy', np.array(loss_cross_l))
    if config['addBC'] == 1:
        np.save(f'{path}/loss_BC_div.npy', np.array(loss_BC_div_l))
        np.save(f'{path}/loss_BC_cul.npy', np.array(loss_BC_cul_l))
    np.save(f'{path}/loss.npy', np.array(loss_l))
    np.save(f'{path}/loss_test.npy', np.array(test_loss_l))

    return best_model
