import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

regularization = 0.0001

def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p))) # L2范数作为正则化项
    return ret

# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1)+y1 for y1 in y_]

def fitting(M=0):
    """
    M    为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='regularization')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq

def leastsquare(x, y, M=0):
    N = len(y)
    xx = np.zeros((N, M+1))
    yy = np.array(y)
    for n in range(N):
        s = 1
        for m in range(M+1):
            xx[n, m] = s
            s *= x[n]
    xs = np.matmul(np.transpose(xx), xx)
    xs = np.linalg.inv(xs)
    xs = np.matmul(xs, np.transpose(xx))
    return np.matmul(xs, yy)

def fitting2(M=0):
    """
    M    为 多项式的次数
    """
    # 最小二乘法
    p_lsq_rev = leastsquare(x, y, M)
    p_lsq = p_lsq_rev[::-1]
    print('Fitting Parameters:', p_lsq)

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq, x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq

# M=0
p_lsq_0 = fitting2(M=0)

# M=1
p_lsq_1 = fitting2(M=1)

# M=3
p_lsq_3 = fitting2(M=3)

# M=9
p_lsq_9 = fitting2(M=9)
p_lsq_9 = fitting(M=9)
