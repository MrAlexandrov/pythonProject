import matplotlib.pyplot as plt
import numpy as np
import pylab as p
from numpy import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from math import *
from scipy import integrate

R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]

Tc = R[1]
Ts = G[1] + B[1] + RGB[2]

l = 1 / Tc      #
m = 1 / Ts      # интенсивность потока обслуживания
p = l / m
A = [1] #       1 / 0!,      p / 1!,              p^2 / 2!,                p^3 / 3!,        p^4 / 4!, ...
P0 = [1] #      1 / 1,  1 / (1 + p),    1 / (1 + p + p^2 / 2!),  1 / (1 + p + p^2 / 2! + p^3 / 3!) ...
# A[i] * P0[i] = P[i]

sm = A[0]

for i in range(1, 100):
    A.append(A[-1] * p / i)
    sm += A[-1]
    P0.append(1 / sm)


def sumAn(n):
    res = 0
    for i in range(n + 1):
        res += A[i]
    return res


n = 12
k = 12

# Q[n][i] = A[n] * (p / n) ^ k
Q = [[0] * (n + 7) for i in range(n + 7)]
P0Q = [[0] * (n + 7) for i in range(n + 7)]


for i in range(n + 7):
    Q[i][0] = 1
    P0Q[i][0] = P0[i]

sm = 0
for i in range(1, n + 5):
    sm = sumAn(i)
    Q[i][0] = A[i]
    for j in range(1, n + 5):
        Q[i][j] = ((p / i) * Q[i][j - 1])
        sm += Q[i][j]
        P0Q[i][j] = 1 / sm


P0INF = [1]
for i in range(1, n + 7):
    sm = sumAn(i)
    a = p / i
    sm += A[i] * a / (1 - a)
    P0INF.append(1 / sm)


def sumQk(n, k):
    res = 0
    for i in range(1, k + 1):
        res += Q[n][i]
    return res


# P0[n] * (A[0] + A[1] + .. + A[n]) = 1
check = sumAn(5)
check *= P0[5]
print(f"check without queue work = {check}")

# P(n) = A[n] * P0[n]
check = sumAn(5)
check *= P0Q[5][0]
print(f"check without queue = {check}")

# P(n + k) = P0Q[n][k] * Q[n][k]
# P0Q[n][k] * (A[0] + A[1] + .. + A[n] +
#             (Q[n][1] + Q[n][2] + .. + Q[n][k])) = 1
# Q[n][k] = (p / n) ^ k * A[n]


check = sumAn(5)
check += sumQk(5, 3)
check *= P0Q[5][3]
print(f"check with queue: {check}")


def check(n, k):
    res = P0Q[n][k] * (sumAn(n) + sumQk(n, k))
    return res


print(f"check(n, k) = {check(5, 3)}")


def Mzan(n):
    x = [i for i in range(6, n + 1)]
    y = []
    for k in x:
        res = 0
        for i in range(k + 1):
            res += P0INF[k] * i * (p ** i) / factorial(i)
        a = p / k
        res += P0INF[k] * (k * (p ** k) / factorial(k)) * (a / (1 - a))
        y.append(res)
    plt.title(f"M зан")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def Kzagr(n):
    x = [i for i in range(6, n + 1)]
    y = []
    for k in x:
        res = 0
        for i in range(k + 1):
            res += P0INF[k] * i * (p ** i) / factorial(i)
        a = p / k
        res += P0INF[k] * (k * (p ** k) / factorial(k)) * (a / (1 - a))
        y.append(res)
    for i in range(1, len(y)):
        y[i] /= i
    plt.title(f"K загр")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k={n}")
    plt.legend()
    plt.show()


def sumPINF(n):
    res = P0INF[n]
    for i in range(1, n):
        res += A[i] * P0INF[n]
    return 1 - res


def Pque(n):
    # x = [i for i in range(k + 1)]
    # y = [sumQk(i, n) for i in x]
    # y[0] = 1
    # # for i in range(1, k + 1):
    # #     y[i] += y[i - 1]
    # for i in range(1, k + 1):
    #     y[i] *= P0Q[i][n]
    x = [i for i in range(1, n)]
    y = [sumPINF(i) for i in x]
    plt.title(f"P оч")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k={n}")
    plt.legend()
    plt.show()


def MlenQ(n):
    x = [i for i in range(k + 1)]
    y = [Q[i][n] * P0Q[i][n] for i in x]
    y[0] = 0
    for i in range(1, k + 1):
        y[i] *= i
        y[i] += y[i - 1]
    plt.title(f"M длины очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k={n}")
    plt.legend()
    plt.show()


def KzanQ(n):
    x = [i for i in range(k + 1)]
    y = [Q[i][n] * P0Q[i][n] for i in x]
    y[0] = 0
    for i in range(1, k + 1):
        y[i] *= i
        y[i] += y[i - 1]
    for i in range(1, k + 1):
        y[i] /= i
    plt.title(f"K занятости очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k={n}")
    plt.legend()


Mzan(n)
Kzagr(n)


