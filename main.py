import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from ConvLayer import ConvLayer
from PoolingLayer import PoolingLayer
from FCLayer import BaseLayer, MiddleLayer, OutputLayer
from cifar10_call import cifar10_call
from mnist_call import mnist_call

import time

start = time.time()
"""초기값 셋팅"""
N = 35000  # How many data Call
eta = 0.00146  # learning rate adam:0.00146
epoch = 20  # learning epoches
batch_size = 32  # Use mini batch
n_sample = 500  # Using for Error, Accuracy samples
alpha = 1 # how many times do you want to learn in 1 epoch. if 1, whole data learns.

input_train, input_test, correct_train, correct_test = mnist_call(N)

n_train = input_train.shape[0]
n_test = input_test.shape[0]

img_h = 28
img_w = 28
img_ch = 1

# -- 각 층의 초기화 --
cl1 = ConvLayer(img_ch, img_h, img_w, 30, 3, 3, stride=1, pad=1)  # 앞3개:인풋 중간3개:필터
cl2 = ConvLayer(cl1.y_ch, cl1.y_h, cl1.y_w, 30, 3, 3, stride=1, pad=1)
pl1 = PoolingLayer(cl2.y_ch, cl2.y_h, cl2.y_w, pool=2, pad=0)  # pool:풀링크기(2*2), pad:패딩 너비
c_dr1 = fn.dropout(0.25)

cl3 = ConvLayer(pl1.y_ch, pl1.y_h, pl1.y_w, 60, 3, 3, stride=1, pad=1)
pl2 = PoolingLayer(cl3.y_ch, cl3.y_h, cl3.y_w, pool=2, pad=0)
c_dr2 = fn.dropout(0.25)

cl4 = ConvLayer(pl2.y_ch, pl2.y_h, pl2.y_w, 120, 3, 3, stride=1, pad=1)

n_fc_in = cl4.y_ch * cl4.y_h * cl4.y_w
ml1 = MiddleLayer(n_fc_in, 500)
dr1 = fn.dropout(0.5)
ml2 = MiddleLayer(500, 500)
dr2 = fn.dropout(0.5)
ol1 = OutputLayer(500, 10)


# -- 순전파--
def forward_propagation(x, is_train=False):
    n_bt = x.shape[0]

    images = x.reshape(n_bt, img_ch, img_h, img_w)
    cl1.forward(images)
    cl2.forward(cl1.y)
    pl1.forward(cl2.y)
    c_dr1.forward(pl1.y, is_train)

    cl3.forward(c_dr1.y)
    pl2.forward(cl3.y)
    c_dr2.forward(pl2.y, is_train)

    cl4.forward(c_dr2.y)

    fc_input = cl4.y.reshape(n_bt, -1)
    ml1.forward(fc_input)
    dr1.forward(ml1.y, is_train)
    ml2.forward(dr1.y)
    dr2.forward(ml2.y, is_train)
    ol1.forward(dr2.y)


# -- 역전파 --
def backpropagation(t):
    n_bt = t.shape[0]

    ol1.backward(t)
    dr2.backward(ol1.grad_x)
    ml2.backward(dr2.grad_x)
    dr1.backward(ml2.grad_x)
    ml1.backward(dr1.grad_x)

    grad_img = ml1.grad_x.reshape(n_bt, cl4.y_ch, cl4.y_h, cl4.y_w)
    cl4.backward(grad_img)

    c_dr2.backward(cl4.grad_x)
    pl2.backward(c_dr2.grad_x)
    cl3.backward(pl2.grad_x)

    c_dr1.backward(cl3.grad_x)
    pl1.backward(c_dr1.grad_x)
    cl2.backward(pl1.grad_x)
    cl1.backward(cl2.grad_x)


# -- 가중치와 편향 수정 --
def uppdate_wb():
    cl1.update(eta)
    cl2.update(eta)
    cl3.update(eta)
    cl4.update(eta)
    ml1.update(eta)
    ml2.update(eta)
    ol1.update(eta)


# -- 오차 계산 --
def get_error(t, batch_size):
    return -np.sum(t * np.log(ol1.y + 1e-7)) / batch_size  # 교차 엔트로피 오차


# -- 샘플을 순전파 --
def forward_sample(inp, correct, n_sample):
    index_rand = np.arange(len(correct))
    np.random.shuffle(index_rand)
    index_rand = index_rand[:n_sample]
    x = inp[index_rand, :]
    t = correct[index_rand, :]
    forward_propagation(x)
    return x, t


# -- 오차 기록용 --
train_error_x = []
train_error_y = []
train_accu_x = []
train_accu_y = []
test_error_x = []
test_error_y = []
test_accu_x = []
test_accu_y = []

# -- 학습과 경과 기록 --
n_batch = n_train // batch_size
if n_batch < alpha:
    print("'alpha' variable is too big")
    exit()
for i in range(epoch):
    es = time.time()
    # train 오차측정
    x, t = forward_sample(input_train, correct_train, n_sample)
    error_train = get_error(t, n_sample)
    # train 정답률측정
    ok, nope = 0, 0
    for k in range(n_sample):
        if np.argmax(t[k, :]) == np.argmax(ol1.y[k, :]):
            ok = ok + 1
        else:
            nope = nope + 1
    accu = 100 * ok / (ok + nope)
    # 테스트 오차측정
    x, t = forward_sample(input_test, correct_test, n_sample)
    error_test = get_error(t, n_sample)
    # 테스트 정답률측정
    ok2, nope2 = 0, 0
    for k in range(n_sample):
        if np.argmax(t[k, :]) == np.argmax(ol1.y[k, :]):
            ok2 = ok2 + 1
        else:
            nope2 = nope2 + 1
    accu2 = 100 * ok2 / (ok2 + nope2)

    # -- 오차 기록 --
    train_error_x.append(i)
    train_error_y.append(error_train * 25)
    train_accu_x.append(i)
    train_accu_y.append(accu)
    test_error_x.append(i)
    test_error_y.append(error_test * 25)
    test_accu_x.append(i)
    test_accu_y.append(accu2)
    print("Epoch:" + str(i) + "/" + str(epoch),
          "Error_train:" + str(error_train), "Error_test:" + str(error_test),
          "Accu_train:" + str(accu), "Accu_test:" + str(accu2))

    # -- 학습 --
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)
    for j in range(n_batch // alpha):
        mb_index = index_rand[j * batch_size: (j + 1) * batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]

        forward_propagation(x, is_train=True)
        backpropagation(t)
        uppdate_wb()
    print("1세대당 ", time.time() - es)
print("time :", time.time() - start)
# -- 오차의 기록을 그래프로 표시 --
plt.plot(train_error_x, train_error_y, label="Train error")
plt.plot(test_error_x, test_error_y, label="Test error")
plt.plot(train_accu_x, train_accu_y, label="Train accu")
plt.plot(test_accu_x, test_accu_y, label="Test accu")
plt.legend()
plt.ylim(0, 100)
plt.grid(b=True, which='both', axis='both')
plt.xlabel("Epochs")
plt.show()
