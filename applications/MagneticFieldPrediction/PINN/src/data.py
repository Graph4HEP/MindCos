import numpy as np
import mindspore as ms
from mindspore import Tensor
import os
import scipy.special as sc


class data_generation:
    def __init__(self, N_sample=3, N_test=1000, L=0.5, dx=2, dy=2, dz=2, a=1, b=1, Ix=1, Iy=1, Iz=1, radius1=1, radius2=1, flip=-1):
        self.N_sample = N_sample
        self.N_test = N_test
        self.L = L
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.a = a
        self.b = b
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.radius1 = radius1
        self.radius2 = radius2
        self.mu0 = 4 * np.pi * 10**-7
        self.Nx = 1
        self.Ny = 1
        self.Nz = 1
        self.flip = -1
    
    def lineB(self, pos, start, end):
        r12 = end - start
        r10 = pos - start
        r20 = pos - end
        cos1 = np.dot(r12, r10) / (np.linalg.norm(r12) * np.linalg.norm(r10))
        cos2 = np.dot(r12, r20) / (np.linalg.norm(r12) * np.linalg.norm(r20))
        r = np.linalg.norm(np.cross(r10, r20)) / np.linalg.norm(r12)
        B = np.cross(r10, r20)
        B = B / np.linalg.norm(B) / r * (cos1 - cos2) / 4 / np.pi
        return B

    def recB_xy(self, x, y, z):
        pos1 = np.array([self.a / 2, self.b / 2, 0])
        pos2 = np.array([-self.a / 2, self.b / 2, 0])
        pos3 = np.array([-self.a / 2, -self.b / 2, 0])
        pos4 = np.array([self.a / 2, -self.b / 2, 0])
        pos = np.array([x, y, z]).squeeze()
        B = self.lineB(pos, pos1, pos2) + self.lineB(pos, pos2, pos3) + self.lineB(pos, pos3, pos4) + self.lineB(pos, pos4, pos1)
        return B

    def circB_rotate(self, x, y, z, theta, phi, radius):
        x_prime = x * np.cos(theta) * np.cos(phi) + y * np.cos(theta) * np.sin(phi) - z * np.sin(theta)
        y_prime = -x * np.sin(phi) + y * np.cos(phi)
        z_prime = x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(theta)
        r_sq_prime = x_prime ** 2 + y_prime ** 2 + z_prime ** 2
        rho_sq_prime = x_prime ** 2 + y_prime ** 2
        alpha_sq_prime = radius ** 2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime) * radius
        beta_sq_prime = radius ** 2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime) * radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        Bx_prime = x_prime * z_prime / (2 * np.pi *alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * (
                (radius ** 2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2 * np.pi * alpha_sq_prime * np.sqrt(beta_sq_prime)) * (
                (radius ** 2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)
        B_prime = np.array([Bx_prime, By_prime, Bz_prime])
        Bx = B_prime[0] * np.cos(theta) * np.cos(phi) - B_prime[1] * np.sin(phi) + B_prime[2] * np.sin(theta) * np.cos(phi)
        By = B_prime[0] * np.cos(theta) * np.sin(phi) + B_prime[1] * np.cos(phi) + B_prime[2] * np.sin(theta) * np.sin(phi)
        Bz = -B_prime[0] * np.sin(theta) + B_prime[2] * np.cos(theta)
        return np.array([Bx, By, Bz])

    def reccircB(self, x, y, z):
        field = 0
        field += self.circB_rotate(x, y + self.dy / 2, z, np.pi / 2, np.pi / 2, self.radius1) * self.flip * self.Iy * self.mu0 * self.Ny
        field += self.circB_rotate(x, y - self.dy / 2, z, np.pi / 2, np.pi / 2, self.radius1) * self.Iy * self.mu0 * self.Ny
        field += self.circB_rotate(x + self.dx / 2, y, z, np.pi / 2, 0, self.radius2) * self.flip * self.Ix * self.mu0 * self.Nx
        field += self.circB_rotate(x - self.dx / 2, y, z, np.pi / 2, 0, self.radius2) * self.Ix * self.mu0 * self.Nx
        field += np.expand_dims(self.recB_xy(x, y, z + self.dz / 2), 1) * self.flip * self.Iz * self.mu0 * self.Nz
        field += np.expand_dims(self.recB_xy(x, y, z - self.dz / 2), 1) * self.Iz * self.mu0 * self.Nz
        return field.tolist()
    
    def train_data(self):
        L1 = self.L
        N = self.N_sample
        x = np.concatenate((-L1 * np.ones([N, 1]), L1 * np.ones([N, 1]), 
                            np.random.default_rng().uniform(low=-L1, high=L1, size=(4 * N, 1))),
                           axis=0)
        y = np.concatenate((np.random.default_rng().uniform(low=-L1, high=L1, size=(2 * N, 1)),
                            -L1 * np.ones([N, 1]), L1 * np.ones([N, 1]),
                            np.random.default_rng().uniform(low=-L1, high=L1, size=(2 * N, 1))),
                           axis=0)
        z = np.concatenate((np.random.default_rng().uniform(low=-L1, high=L1, size=(4 * N, 1)),
                            -L1 * np.ones([N, 1]), L1 * np.ones([N, 1])),
                           axis=0)

        x = Tensor(x, ms.float32)
        y = Tensor(y, ms.float32)
        z = Tensor(z, ms.float32)
        pos = ms.ops.Concat(1)([x, y, z])
        labels = Tensor(np.array([self.reccircB(x[i].asnumpy(), y[i].asnumpy(), z[i].asnumpy()) for i in range(len(x))]), ms.float32)
        return pos, labels

    def test_data(self):
        L = self.L
        N = self.N_test
        x = np.random.default_rng().uniform(low=-L, high=L, size=((N, 1)))
        y = np.random.default_rng().uniform(low=-L, high=L, size=((N, 1)))
        z = np.random.default_rng().uniform(low=-L, high=L, size=((N, 1)))

        x = Tensor(x, ms.float32)
        y = Tensor(y, ms.float32)
        z = Tensor(z, ms.float32)
        pos = ms.ops.Concat(1)([x, y, z])

        labels = Tensor(np.array([self.reccircB(x[i].asnumpy(), y[i].asnumpy(), z[i].asnumpy()) for i in range(len(x))]), ms.float32)
        return pos, labels

    def single_point(self, x, y, z):
        x = Tensor(x, ms.float32).reshape(-1,1)
        y = Tensor(y, ms.float32).reshape(-1,1)
        z = Tensor(z, ms.float32).reshape(-1,1)
        pos = ms.ops.Concat(1)([x, y, z])
        labels = Tensor(np.array([self.reccircB(x[i].asnumpy(), y[i].asnumpy(), z[i].asnumpy()) for i in range(len(x))]), ms.float32)
        return pos, labels*1e6
