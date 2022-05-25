"""
单纯形算法
标准形式 minf(x)
        Ax<=b
        x>=0
    max*-1变min
    Ax>=b两边乘上*-1变Ax<=b
"""
import numpy as np
import time

class Simplex(object):
    def __init__(self, obj, max_mode=False):  # 默认min, 若max乘上-1
        self.mat, self.max_mode = np.array([[0] + obj]) * (-1 if max_mode else 1), max_mode

    def add_constraint(self, a, b):
        """
         根据方程添加松弛变量
        :param a: 约束方程左边的系数
        :param b: 约束方程右边的系数
        :mat: 标准化的约束方程
        """
        self.mat = np.vstack([self.mat, [b] + a])

    def simplex_(self, mat, B, m, n):
        while mat[0, 1:].min() < 0:
            col = np.where(mat[0, 1:] < 0)[0][0] + 1  # 采用Bland规则避免退化
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # 获取theta对应的索引
            if mat[row][col] <= 0:return None  # 如果theta为无穷，即A系数为0，那么无解
            self.pivot_(mat, B, row, col)
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n}

    def pivot_(self, mat, B, row, col):
        mat[row] /= mat[row][col]
        ids = np.arange(mat.shape[0]) != row
        mat[ids] -= mat[row] * mat[ids, col:col + 1]
        B[row] = col


    def solve(self):
        m, n = self.mat.shape  # 添加的松弛变量个数为m-1个
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]), list(range(n - 1, n + m - 1))
        mat = self.mat = np.hstack([self.mat, temp])
        if mat[1:, 0].min() < 0:  # 判断初始解是否为基可行解？
            row = mat[1:, 0].argmin() + 1  # 获取最小b的索引
            temp, mat[0] = np.copy(mat[0]), 0
            mat = np.hstack([mat, np.array([1] + [-1] * (m - 1)).reshape((-1, 1))])
            self.pivot_(mat, B, row, mat.shape[1] - 1)
            if self.simplex_(mat, B, m, n)[0] != 0: return None  # 无解

            if mat.shape[1] - 1 in B:
                self.pivot_(mat, B, B.index(mat.shape[1] - 1), np.where(mat[0, 1:] != 0)[0][0] + 1)
            self.mat = np.vstack([temp, mat[1:, :-1]])
            for i, x in enumerate(B[1:]):
                self.mat[0] -= self.mat[0, x] * self.mat[i + 1]
        return self.simplex_(self.mat, B, m, n)

# 输入方程
t = Simplex([3, 9, 20, 19])

# 以小于等于为标准形式，如果约束方程为大于等于需两边同乘-1
t.add_constraint([-110, -160, -420, -260], -2000)
t.add_constraint([-4, -8, -4, -14], -55)
t.add_constraint([-2, -285, -22, -80], -800)

start=time.time()
print('单纯形法解线性规划问题得到的解：\t',t.solve())
print(t.mat)
end=time.time()
print("程序的运行时间：{}".format(end-start))