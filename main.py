
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from ConvLayer import ConvLayer
from PoolingLayer import PoolingLayer
from FCLayer import BaseLayer, MiddleLayer, OutputLayer
from mnist_call import mnist_call


N=2000
eta = 0.01
epoch = 30
batch_size = 200
n_sample = 200
# 학습데이터 호출
images_train, images_test, labels_train, labels_test = mnist_call(N)


# 사용에 쓰이는 data로 변환완료
input_train = images_train
correct_train = labels_train
input_test = images_test
correct_test = labels_test

n_train = input_train.shape[0]
n_test  = input_test.shape[0]

"""초기값 셋팅"""
img_h = 28
img_w = 28
img_ch = 1

# -- 각 층의 초기화 --
cl_1 = ConvLayer(img_ch, img_h, img_w, 10, 5, 5, stride = 1, pad = 2) #앞3개:인풋 중간3개:필터
pl_1 = PoolingLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, pool = 2, pad = 0) # pool:풀링 영역 크기, pad:패딩 너비

n_fc_in = pl_1.y_ch * pl_1.y_h * pl_1.y_w
ml_1 = MiddleLayer(n_fc_in, 100)
ol_1 = OutputLayer(100, 10)


# -- 순전파--
def forward_propagation(x):
    n_bt = x.shape[0]

    images = x.reshape(n_bt, img_ch, img_h, img_w)
    cl_1.forward(images)
    pl_1.forward(cl_1.y)

    fc_input = pl_1.y.reshape(n_bt, -1)
    ml_1.forward(fc_input)
    ol_1.forward(ml_1.y)


# -- 역전파 --
def backpropagation(t):
    n_bt = t.shape[0]

    ol_1.backward(t)
    ml_1.backward(ol_1.grad_x)

    grad_img = ml_1.grad_x.reshape(n_bt, pl_1.y_ch, pl_1.y_h, pl_1.y_w)
    pl_1.backward(grad_img)
    cl_1.backward(pl_1.grad_x)


# -- 가중치와 편향 수정 --
def uppdate_wb():
    cl_1.update(eta)
    ml_1.update(eta)
    ol_1.update(eta)


# -- 오차 계산 --
def get_error(t, batch_size):
    return -np.sum(t * np.log(ol_1.y + 1e-7)) / batch_size  # 교차 엔트로피 오차


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
train_error_x = []; train_error_y = []
train_accu_x = []; train_accu_y = []
test_error_x = []; test_error_y = []
test_accu_x = []; test_accu_y = []


# -- 학습과 경과 기록 --
n_batch = n_train // batch_size
for i in range(epoch) :

    # train 오차측정
    x, t = forward_sample(input_train, correct_train, n_sample)
    error_train = get_error(t, n_sample)
    # train 정답률측정
    ok, nope = 0, 0
    for k in range(n_sample) :
        if np.argmax(t[k,:]) == np.argmax(ol_1.y[k,:]) :
            ok = ok + 1
        else :
            nope = nope + 1
    accu = 100*ok/(ok+nope)
    # 테스트 오차측정
    x, t = forward_sample(input_test, correct_test, n_sample)
    error_test = get_error(t, n_sample)
    # 테스트 정답률측정
    ok2, nope2 = 0, 0
    for k in range(n_sample):
        if np.argmax(t[k, :]) == np.argmax(ol_1.y[k, :]):
            ok2 = ok2 + 1
        else:
            nope2 = nope2 + 1
    accu2 = 100 * ok2 / (ok2 + nope2)

    # -- 오차 기록 --
    train_error_x.append(i)
    train_error_y.append(error_train)
    train_accu_x.append(i)
    train_accu_y.append(accu)
    test_error_x.append(i)
    test_error_y.append(error_test)
    test_accu_x.append(i)
    test_accu_y.append(accu2)

    # -- 경과 표시 --

    print("Epoch:" + str(i) + "/" + str(epoch), "Error_train:" + str(error_train), "Error_test:" + str(error_test),
            "Accu_train:" + str(accu), "Accu_test:" + str(accu2))

    # -- 학습 --
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)
    for j in range(n_batch):
        mb_index = index_rand[j * batch_size: (j + 1) * batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]

        forward_propagation(x)
        backpropagation(t)
        uppdate_wb()

    # -- 오차의 기록을 그래프로 표시 --
plt.plot(train_error_x, train_error_y, label="Train error")
plt.plot(test_error_x, test_error_y, label="Test error")
plt.plot(train_accu_x, train_accu_y, label="Train accu")
plt.plot(test_accu_x, test_accu_y, label="Test acuu")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()