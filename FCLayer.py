import numpy as np
import functions as fn
from scipy.special import softmax

# -- 전결합층의 부모 클래스 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = 0.1*np.random.randn(n_upper, n)
        self.b = 0.1*np.random.randn(n)
        #adam
        self.m_w = np.zeros((n_upper, n))
        self.v_w = np.zeros((n_upper, n))
        self.m_b = np.zeros(n)
        self.v_b = np.zeros(n)
        self.t = 1

    def update(self, eta):
        #adam
        self.m_w = (0.9 * self.m_w) + (0.1 * self.grad_w)
        self.v_w = (0.999 * self.v_w) + (0.001 * (self.grad_w * self.grad_w))
        m = self.m_w / (1 - (0.9 ** self.t))
        v = self.v_w / (1 - (0.999 ** self.t))
        self.w -= (eta * m) / (np.sqrt(v) + 1e-8)

        self.m_b = (0.9 * self.m_b) + (0.1 * self.grad_b)
        self.v_b = (0.999 * self.v_b) + (0.001 * (self.grad_b * self.grad_b))
        m2 = self.m_b / (1 - (0.9 ** self.t))
        v2 = self.v_b / (1 - (0.999 ** self.t))
        self.b -= (eta * m2) / (np.sqrt(v2) + 1e-8)
        self.t += 1




# -- 전결합 은닉층 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = x @ self.w + self.b
        self.y = fn.relu(self.u)

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)

        self.grad_w = self.x.T @ delta
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = delta @ self.w.T

    # -- 전결합 출력층 --


class OutputLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        u = x @ self.w + self.b
        self.y = softmax(u, axis=1)

    def backward(self, t):
        delta = self.y - t

        self.grad_w = self.x.T @ delta
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = delta @ self.w.T
