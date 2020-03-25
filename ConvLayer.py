import numpy as np
import functions as fn

class ConvLayer :
    def __init__(self, x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride=1, pad=1) :
        self.params = (x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad)
        #가중치, 바이어스 초기값
        self.w =  0.1*np.random.randn(n_flt, x_ch, flt_h, flt_w)
        self.b =  0.1*np.random.randn(1, n_flt)

        self.y_ch = n_flt
        self.y_h  = (x_h - flt_h + 2*pad) // stride + 1
        self.y_w  = (x_w - flt_w + 2*pad) // stride + 1
        #adagrad & rmsprop
        # self.h_w = np.zeros((n_flt, x_ch, flt_h, flt_w)) + 1e-8
        # self.h_b = np.zeros((1, n_flt)) + 1e-8
        # adam
        self.m_w = np.zeros((n_flt, x_ch, flt_h, flt_w))
        self.v_w = np.zeros((n_flt, x_ch, flt_h, flt_w))
        self.m_b = np.zeros((1, n_flt))
        self.v_b = np.zeros((1, n_flt))
        self.t = 1

    def forward(self, x) :
        n_bt = x.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        #input,필터 전처리(im2col)
        self.cols = fn.im2col(x, flt_h, flt_w, y_h, y_w, stride=stride, pad=pad)
        self.w_col = self.w.reshape(n_flt, x_ch * flt_h * flt_w)
        #출력
        u = (self.w_col @ self.cols).T + self.b
        self.u = u.reshape(n_bt, y_h, y_w, y_ch).transpose(0,3,1,2)
        self.y = fn.relu(self.u)

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # delta
        delta = grad_y * np.where(self.u <= 0, 0, 1) #0이하이면 0, 아니면 1(ReLU Diff)
        delta = delta.transpose(0, 2, 3, 1).reshape(n_bt * y_h * y_w, y_ch)

        # 필터와 편향 기울기
        grad_w = self.cols @ delta
        self.grad_w = grad_w.T.reshape(n_flt, x_ch, flt_h, flt_w)
        self.grad_b = np.sum(delta, axis=0)

        # 입력 기울기
        grad_cols = delta @ self.w_col
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = fn.col2im(grad_cols.T, x_shape, flt_h, flt_w, y_h, y_w, stride, pad)

    def update(self, eta):
        #adagrad
        # self.h_w += self.grad_w * self.grad_w
        # self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        #
        # self.h_b += self.grad_b * self.grad_b
        # self.b -= eta / np.sqrt(self.h_b) * self.grad_b
        #rmsprop
        # self.h_w = (0.9 * self.h_w) + (0.1 * (self.grad_w * self.grad_w))
        # self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        #
        # self.h_b =(0.9 * self.h_b) + (0.1 * (self.grad_b * self.grad_b))
        # self.b -= eta / np.sqrt(self.h_b) * self.grad_b
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