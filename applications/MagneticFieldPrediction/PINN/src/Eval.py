import mindspore as ms
from mindspore import Tensor
import matplotlib.pyplot as plt
import time
import numpy as np

def getdB(Bpred, Breal):
    Bx = Bpred[:, 0].asnumpy()
    By = Bpred[:, 1].asnumpy()
    Bz = Bpred[:, 2].asnumpy()
    Bx_r = Breal[:, 0].asnumpy()
    By_r = Breal[:, 1].asnumpy()
    Bz_r = Breal[:, 2].asnumpy()
    dBx_rel = (Bx - Bx_r) / Bx_r
    dBy_rel = (By - By_r) / By_r
    dBz_rel = (Bz - Bz_r) / Bz_r
    dB_rel = (np.sqrt(Bx**2 + By**2 + Bz**2) - np.sqrt(Bx_r**2 + By_r**2 + Bz_r**2)) / np.sqrt(Bx_r**2 + By_r**2 + Bz_r**2)

    dBx = Bx - Bx_r
    dBy = By - By_r
    dBz = Bz - Bz_r
    dB = np.sqrt(Bx**2 + By**2 + Bz**2) - np.sqrt(Bx_r**2 + By_r**2 + Bz_r**2)

    return dBx, dBy, dBz, dB, dBx_rel, dBy_rel, dBz_rel, dB_rel

def Eval(model, config, field):
    st = time.time()
    path = config['path']
    mode = config['geo']
    Btype = config['Btype']
    mean = config['mean']
    std = config['std']
    L = config['length'] / 2
    maxi = config['maxi']
    N_val = 101

    model.to(config['device'])
    if mode == 'cube':
        x_test_np_grid = np.linspace(-L, L, N_val)
    if mode == 'slice':
        x_test_np_grid = np.linspace(-0, 0, N_val)
    y_test_np_grid = np.linspace(-L, L, N_val)
    z_test_np_grid = np.linspace(-L, L, N_val)
    xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False)
    x_test_np = xx.reshape((N_val**3, 1))
    y_test_np = yy.reshape((N_val**3, 1))
    z_test_np = zz.reshape((N_val**3, 1))

    x_test = Tensor(x_test_np, dtype=ms.float32)
    y_test = Tensor(y_test_np, dtype=ms.float32)
    z_test = Tensor(z_test_np, dtype=ms.float32)
    inputs = ms.ops.Concat(axis=1)((x_test, y_test, z_test)) / maxi

    if Btype == 'Helmholtz':
        temp_final = np.array([field.HelmholtzB(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
    elif Btype == 'normal':
        temp_final = np.array([field.B(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
    elif Btype == 'reccirc':
        temp_final = np.array([field.reccircB(x_test_np[i], y_test_np[i], z_test_np[i]) for i in range(N_val**3)])
    temp_final = temp_final.reshape((N_val, N_val, N_val, 3))
    inputs = inputs.to(config['device'])
    std = Tensor(std, dtype=ms.float32).to(config['device'])
    mean = Tensor(mean, dtype=ms.float32).to(config['device'])
    model_output = model(inputs).view(N_val, N_val, N_val, 3) * std + mean

    model_output = model_output.reshape(N_val**3, 3).asnumpy()
    temp_final = temp_final.reshape(N_val**3, 3)
    print('calculation done!')

    dBx, dBy, dBz, dB, dBx_rel, dBy_rel, dBz_rel, dB_rel = getdB(Tensor(model_output), Tensor(temp_final))
    fig_stat = plt.figure(figsize=([32, 16]))
    # ... (省略中间的绘图代码) ...
    plt.savefig(f'{path}/hist_result.png')
    plt.show()
    plt.close()

    model_output = model_output.reshape(N_val, N_val, N_val, 3)
    temp_final = temp_final.reshape(N_val, N_val, N_val, 3)

    ax = ['Bx', 'By', 'Bz']
    # ... (省略中间的绘图代码) ...

if __name__ == '__main__':
    import sys
    import json
    path = sys.argv[1]
    with open(f'{path}/config.json', 'r') as f:
        config = json.load(f)
    print(config)
    model = PINN(config['units'])
    model.load_parameter(f'{path}/best_model.pt')
    field = data_generation(radius=config['radius'],
                            N_sample=config['Nsamples'],
                            N_test=config['Ntest'],
                            L=config['length'] / 2,
                            dx=config['dx'],
                            dy=config['dy'],
                            dz=config['dz'],
                            radius1=config['radius1'],
                            radius2=config['radius2'],
                            a=config['a'],
                            b=config['b'],
                            Iz=config['Iz'],
                            Ix=config['Ix'],
                            Iy=config['Iy']
                            )
    Eval(model, config, field)
