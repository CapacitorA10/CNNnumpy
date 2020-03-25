
import numpy as np
import tensorflow as tf

def mnist_call(a):
    """ MNIST DATASET 불러오기"""
    batch_size = a
    test_batch_size = batch_size // 6

    data_train, data_test        = tf.keras.datasets.mnist.load_data()
    (images_train, labels_train) = data_train
    (images_test, labels_test)   = data_test

    #batch_size*28*28 크기로 배치크기 조절
    # images
    #images_validation = images_train[-batch_size:]
    images_train      = images_train[:batch_size]
    images_test       = images_test[:test_batch_size]

    # labels
    # labels_validation = labels_train[-batch_size:]
    labels_train = labels_train[:batch_size]
    labels_test = labels_test[:test_batch_size]

    #one hot encoding
    k = np.zeros((batch_size, 10), dtype=int)
    for i in range(batch_size) :
        k[i, labels_train[i]] = 1
    del(labels_train)
    labels_train = k

    k2 = np.zeros((test_batch_size, 10), dtype=int)
    for i in range(test_batch_size):
        k2[i, labels_test[i]] = 1
    del(labels_test)
    labels_test = k2



    # (B,C, (H*W))로 수정
    images_train = images_train.reshape(batch_size, -1)
    images_test = images_test.reshape(test_batch_size, -1)
    # max = 1로 수정
    images_train = images_train / 255
    images_test  = images_test / 255
    return images_train, images_test, labels_train, labels_test