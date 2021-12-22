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


def que(n):
    x = [i for i in range(1, n + 1)]
    y = [Q[n][i] for i in x]
    plt.title(f"Lоч, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("L")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def check(n, k):
    res = P0Q[n][k] * (sumAn(n) + sumQk(n, k))
    return res


print(f"check(n, k) = {check(5, 3)}")


def otk(n):
    x = [i for i in range(n + 1)]
    y = [P0Q[n][i] * Q[n][i] for i in x]
    # y[0] = 1
    plt.title(f"Pотк, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def busy(n):
    x = [i for i in range(n + 1)]
    y = [A[i] * P0Q[n][i] * i for i in x]
    for i in range(1, n + 1):
        y[i] += y[i - 1]
    plt.title(f"Mзан, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def zagr(n):
    x = [i for i in range(n + 1)]
    y = [A[i] * P0Q[n][i] * i for i in x]
    for i in range(1, n + 1):
        y[i] += y[i - 1]
    for i in range(1, n + 1):
        y[i] /= i
    plt.title(f"Kзагр, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("K")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def Pqueue(n):
    x = [i for i in range(n + 1)]
    y = [Q[n][i] for i in x]
    for i in range(1, n + 1):
        y[i] += y[i - 1]
    for i in range(1, n + 1):
        y[i] *= P0Q[n][i]
    y[0] = 0
    plt.title(f"Pоч, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def Qlen(n):
    x = [i for i in range(n + 1)]
    y = [Q[n][i] for i in x]
    y[0] = 0
    for i in range(1, n + 1):
        y[i] *= i * P0Q[n][i] * A[n]
        y[i] += y[i - 1]
    plt.title(f"M длины очереди, n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()

# for i in range(2, 15, 2):
# n = 3
que(n)
otk(n)
busy(n)
zagr(n)
Pqueue(n)
Qlen(n)

# x = []
# y = []
#
# plt.title("P(3 <= k)")  # заголовок
# plt.xlabel("n")  # ось абсцисс
# plt.ylabel("P(k)")  # ось ординат
# plt.grid()  # включение отображение сетки
# plt.plot(x, y)
# plt.show()
