import numpy as np
import functions as fn
from scipy.special import softmax

# -- 전결합층의 부모 클래스 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = 0.1*np.random.randn(n_upper, n)
        self.b = 0.1*np.random.randn(n)

        self.h_w = np.zeros((n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8

    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w

        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b


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
