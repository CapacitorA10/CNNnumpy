import numpy as np

def im2col(images, flt_h, flt_w, out_h, out_w, stride=1, pad=0) :
    n_bt, n_ch, img_h, img_w = images.shape
    img_pad = np.pad(images, [(0,0), (0,0), (pad,pad), (pad, pad)], "constant")
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h) :
        h_lim = h + stride * out_h
        for w in range(flt_w) :
            w_lim = w + stride * out_w
            cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride, w:w_lim:stride]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch * flt_h * flt_w, n_bt * out_h * out_w)

    return cols


def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride=1, pad=0) :
    n_bt, n_ch, img_h, img_w = img_shape
    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))

    for h in range(flt_h) :
        h_lim = h + stride * out_h
        for w in  range(flt_w) :
            w_lim = w + stride * out_w
            images[:, :, h:h_lim:stride, w:w_lim:stride] += cols[:, :, h, w, :, :]

    return images[:, :, pad:img_h+pad, pad:img_w+pad]

def relu(x):
    return np.maximum(0,x)

class dropout :
    def __init__(self, ratio) :
        self.ratio = ratio

    def forward(self, x, is_train):
        if is_train :
            rand = np.random.rand(*x.shape)
            self.dropout = np.where(rand > self.ratio, 1, 0)
            self.y = x * self.dropout
        else :
            self.y = (1 - self.ratio) * x #없으면 overflow

    def backward(self, grad):
        self.grad_x = grad * self.dropout