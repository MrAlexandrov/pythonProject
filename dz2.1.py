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
p = RED / ALL
q = NERED / ALL


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
    return C(n, k) * (p ** k) * (q ** (n - k))


def X(n, k):
    return (k - n * p) / sqrt(n * p * q)


def MuLa(n, k):
    return fi(X(n, k))


def P(k1, k2, n):
    x1 = X(n, k1)
    x2 = X(n, k2)
    return FI(x2) - FI(x1)


n1 = [6, 9, 12]
n2 = [12]
n3 = [25, 50, 100, 200, 400, 1000]
n4 = [25, 50, 100, 200, 400]
n5 = [1000]
# n = n1 + n2 + n3 + n4


for n in n1:
    x = [i for i in range(n + 1)]
    y = [Bernulli(n, i) for i in range(n + 1)]
    plt.title(f"n = {n}")                         # заголовок
    plt.xlabel("k")                             # ось абсцисс
    plt.ylabel("P(k)")                          # ось ординат
    plt.grid()                                  # включение отображение сетки
    plt.bar(x, y)                            # построение графика
    #plt.scatter(x, y)
    plt.show()

# 2
for n in n2:
    x = [i for i in range(n + 1)]
    y = [Bernulli(n, i) for i in range(n + 1)]
    y1 = y
    for i in range(1, (n + 1)):
        y1[i] += y1[i - 1]

    plt.title(f"n = {n}")  # заголовок
    plt.xlabel("x")  # ось абсцисс
    plt.ylabel("P(k <= x)")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.step(x, y)                            # построение графика
    # plt.scatter(x, y)
    plt.show()


# 1000: 250 - 350
for n in n3:
    # x = [i for i in range(n + 1)]
    # y = [Bernulli(n, i) for i in x]
    global l, r, step
    x = [i for i in range(n + 1)]
    y = []
    if n == 25:
        l = 0
        r = 15
        x = [0, 1, 3, 5, 7, 9, 11, 13, 15]
    elif n == 50:
        l = 5
        r = 25
        x = [0, 4, 7, 9, 11, 12, 14, 15, 17, 18, 20, 22, 24]
    elif n == 100:
        l = 15
        r = 45
        x = [0, 18, 21, 25, 29, 33, 37, 40]
    elif n == 200:
        l = 40
        r = 80
        x = [0, 42, 48, 53, 59, 65, 70, 76]
    elif n == 400:
        l = 90
        r = 150
        x = [0, 93, 102, 109, 118, 127, 134, 143]
    elif n == 1000:
        l = 250
        r = 350
        x = [0, 259, 272, 281, 296, 311, 320, 333]
    l = 0
    r = n + 1
    y = [MuLa(n, i) for i in x]
    x = x[l:r]
    y = y[l:r]
    plt.title(f"n = {n}")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("P(k)")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.bar(x, y)                            # построение графика
    # plt.scatter(x, y)
    plt.show()


# Part 4
# x = [i for i in range(1, 100000)]
for n in n4:
    x = [i for i in range(1, n + 1)]
    y = [2 * FI(RED / sqrt(i * p * q)) for i in x]
    plt.title(f"P(|k - n * p| <= delta), n = {n}")  # заголовок
    plt.xlabel("eps")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)  # построение графика
    plt.show()

# Part 5
eps = [1e-1, 1e-2, 1e-3]
for n in n5:
    # x = arange(1e-1, 1e-4, -1e-4)
    x = eps
    y = [2 * FI(i * sqrt(n / (p * q))) for i in x]
    plt.title(f"P(|k / n - p| <= eps)")  # заголовок
    plt.xlabel("eps")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)  # построение графика
    plt.show()

#Part 6
# Рассчитать допустимый интервал числа успешных испытаний k
# (симметричный относительно математического ожидания),
# обеспечивающий попадание в него с вероятностью P =
# 0,7; 0,8; 0,9; 0,95; 0,99.

t = [0.7, 0.8, 0.9, 0.95, 0.99]
for n in n5:
    for i in t:
        l = 0
        r = n
        while r - l > 1:
            m = (l + r) // 2
            if i < 2 * FI(m / sqrt(n * p * q)):
                r = m
            else:
                l = m
        print(f"P(M - {r} <= k <= M + {r}) = {i}")
        print(FI(r / sqrt(n * p * q)) - FI( - r / sqrt(n * p * q)))

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

plt.title(f"Min n for P(k <= n <= inf)")                  # заголовок
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