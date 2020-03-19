import numpy as np
import functions as fn

# -- 풀링층 --
class PoolingLayer:

    # n_bt:배차 사이즈, x_ch:입력 채널 수, x_h:입력 이미지 높이, x_w:입력 이미지 너비
    # pool:풀링 영역 크기, pad:패딩 너비
    # y_ch:출력 채널 수, y_h:출력 높이, y_w:출력 너비

    def __init__(self, x_ch, x_h, x_w, pool = 2, pad = 0):
        # 파라미터 정리
        self.params = (x_ch, x_h, x_w, pool, pad)

        # 출력 이미지 크기
        self.y_ch = x_ch  # 출력 채널 수
        self.y_h = x_h // pool if x_h % pool == 0 else x_h // pool + 1  # 출력 높이
        self.y_w = x_w // pool if x_w % pool == 0 else x_w // pool + 1  # 출력 너비

    def forward(self, x):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 입력 이미지를 행렬로 변환
        cols = fn.im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt * y_h * y_w * x_ch, pool * pool)

        # 출력 계산: 맥스풀링
        y = np.max(cols, axis=1)
        self.y = y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)

        # 최대값 인덱스 저장
        self.max_index = np.argmax(cols, axis=1)

    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 출력 기울기의 축 변경
        grad_y = grad_y.transpose(0, 2, 3, 1)

        # 행렬을 생성하고、각 열의 최대값이 있던 위치에 출력 기울기 입력
        grad_cols = np.zeros((pool * pool, grad_y.size))
        grad_cols[self.max_index.reshape(-1), np.arange(grad_y.size)] = grad_y.reshape(-1)
        grad_cols = grad_cols.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        grad_cols = grad_cols.transpose(5, 0, 1, 2, 3, 4)
        grad_cols = grad_cols.reshape(y_ch * pool * pool, n_bt * y_h * y_w)

        # 입력 기울기
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = fn.col2im(grad_cols, x_shape, pool, pool, y_h, y_w, pool, pad)
