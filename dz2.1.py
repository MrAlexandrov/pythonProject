import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from math import *
from scipy import integrate

R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]
RED = R[1]
ALL = RGB[1]
NERED = ALL - RED


def fi(t):
    return 1.0 / sqrt(2 * pi) * exp((-t ** 2) / 2)


def FI(t):
    otr = 1
    if t < 0:
        otr = -1
        t = -t
    y = lambda x: fi(x)  # 1.0 / sqrt(2 * pi) * exp((-x ** 2) / 2)
    res = integrate.quad(y, 0, t)
    return otr * res[0]


def C(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


def Bernulli(n, k):
    p = RED / ALL
    q = NERED / ALL
    return C(n, k) * (p ** k) * (q ** (n - k))


def P(k1, k2, n):
    p = RED / ALL
    q = NERED / ALL
    x1 = (k1 - n * p) / sqrt(n * p * q)
    x2 = (k2 - n * p) / sqrt(n * p * q)
    return FI(x2) - FI(x1)


n1 = [6, 9, 12]
n2 = [12]
n3 = [25, 50, 100, 200, 400, 1000]
n4 = [25, 50, 100, 200, 400]
n5 = [1000]
# n = n1 + n2 + n3 + n4


# for n in n1:
#     x = [i for i in range(n + 1)]
#     y = [Bernulli(n, i) for i in range(n + 1)]
#     plt.title(f"n = {n}")                         # заголовок
#     plt.xlabel("k")                             # ось абсцисс
#     plt.ylabel("P(k)")                          # ось ординат
#     plt.grid()                                  # включение отображение сетки
#     # plt.plot(x, y)                            # построение графика
#     plt.scatter(x, y)
#     plt.show()
#
# for n in n2:
#     x = [i for i in range(n + 1)]
#     y = [Bernulli(n, i) for i in range(n + 1)]
#     y1 = y
#     for i in range(1, (n + 1)):
#         y1[i] += y1[i - 1]
#
#     plt.title(f"n = {n}")  # заголовок
#     plt.xlabel("x")  # ось абсцисс
#     plt.ylabel("P(k <= x)")  # ось ординат
#     plt.grid()  # включение отображение сетки
#     plt.plot(x, y)                            # построение графика
#     # plt.scatter(x, y)
#     plt.show()


# 1000: 250 - 350
# for n in n3:
#     x = [i for i in range(n + 1)]
#     y = [Bernulli(n, i) for i in x]
#     plt.title(f"n = {n}")  # заголовок
#     plt.xlabel("k")  # ось абсцисс
#     plt.ylabel("P(k)")  # ось ординат
#     plt.grid()  # включение отображение сетки
#     plt.plot(x, y)                            # построение графика
#     # plt.scatter(x, y)
#     plt.show()


# Part 4
# # x = [i for i in range(1, 100000)]
# x = n4
# y = []
# p = RED / ALL
# q = NERED / ALL
# for i in x:
#     y.append(2 * FI(RED / sqrt(i * p * q)))


# Part 5
eps = [1e-1, 1e-2, 1e-3]
for n in n5:
    p = RED / ALL
    q = NERED / ALL
    # x = arange(1e-1, 1e-4, -1e-4)
    # x = eps
    # y = [2 * FI(i * sqrt(n / (p * q))) for i in x]
    # plt.title(f"|k / n - p| <= eps")  # заголовок
    # plt.xlabel("eps")  # ось абсцисс
    # plt.ylabel("P(..)")  # ось ординат
    # plt.grid()  # включение отображение сетки
    # plt.plot(x, y)  # построение графика
    # plt.show()

#Part 6
# Рассчитать допустимый интервал числа успешных испытаний k
# (симметричный относительно математического ожидания),
# обеспечивающий попадание в него с вероятностью P =
# 0,7; 0,8; 0,9; 0,95; 0,99.

# t = [0.7, 0.8, 0.9, 0.95, 0.99]
# for n in n5:
#     for i in t:
#         l = 0
#         r = n
#         while r - l > 1:
#             m = (l + r) // 2
#             if i < 2 * FI(m / sqrt(n * p * q)):
#                 r = m
#             else:
#                 l = m
#         print(f"P(M - {r} <= k <= M + {r}) = {i}")
#         print(FI(r / sqrt(n * p * q)) - FI( - r / sqrt(n * p * q)))

#Part 7
# Построить график зависимости минимально необходимого числа испытаний n,
# для того чтобы обеспечить вероятность появления не менее, чем
# N1 = R1 + G1 + B1 красных шаров с вероятностями
# P = 0.7, 0.8, 0.9, 0.95, 0.99

pp = [0.7, 0.8, 0.9, 0.95, 0.99]
k = RGB[1]
x = [i for i in range(k, 150)]
# x = [k, 73, 78, 86, 92, 103, 200]
y = [P(k, i, i) for i in x]
# for i in range(1, len(y)):
#     y[i] += y[i - 1]

for i in pp:
    for j in range(len(y)):
        if y[j] >= i:
            print(f"Min n for P = {i} is {j}")
            break

plt.title(f"Min n for P(...)")                  # заголовок
plt.xlabel("n")                                 # ось абсцисс
plt.ylabel("P")                                 # ось ординат
plt.grid()                                      # включение отображение сетки
plt.plot(x, y)                                  # построение графика
plt.show()


# Независимая (x) и зависимая (y) переменные
# x = np.linspace(0, 10, 50)
# y1 = x
# y2 = [i**2 for i in x]
#
#                                             # Построение графика
# plt.title("Линейная зависимость y = x")     # заголовок
# plt.xlabel("x")                             # ось абсцисс
# plt.ylabel("y1, y2")                        # ось ординат
# plt.grid()                                  # включение отображение сетки
# plt.plot(x, y1, x, y2)                      # построение графика
# plt.show()


# x = np.linspace(0, 10, 10)
# y1 = 4*x
# y2 = [i**2 for i in x]
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_title("Графики зависимостей: y1=4*x, y2=x^2", fontsize=16)
# ax.set_xlabel("x", fontsize=14)
# ax.set_ylabel("y1, y2", fontsize=14)
# ax.grid(which="major", linewidth=1.2)
# ax.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
# ax.scatter(x, y1, c="red", label="y1 = 4*x")
# ax.plot(x, y2, label="y2 = x^2")
# ax.legend()
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())
# ax.tick_params(which='major', length=10, width=2)
# ax.tick_params(which='minor', length=5, width=1)
# plt.show()


# fruits = ["apple", "peach", "orange", "bannana", "melon"]
# counts = [34, 25, 43, 31, 17]
# plt.bar(fruits, counts)
# plt.title("Fruits!")
# plt.xlabel("Fruit")
# plt.ylabel("Count")
# plt.show()

# plot - график
# bar - столбчатая диаграмма