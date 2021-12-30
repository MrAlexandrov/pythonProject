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
ALL = RGB[1] + RGB[2] + RGB[3] + 1
GOOD = 1
BAD = ALL - GOOD
EXP = [100, 1000, 10000]                # Число экспериментов
p = GOOD / ALL
q = BAD / ALL


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


def Stir(n):
    return sqrt(2 * pi) * (n / exp(1)) ** n * exp(1 / (12 * n + 0.7509))


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


def MathWait(P):
    res = 0
    for i in range(len(P)):
        res += i * P[i]
    return res


# D(x) = f(x) * (x - M) ^ 2
def Disp1(x, P):
    res = 0
    M = MathWait(P)
    for i in range(len(P)):
        res += P[i] * ((x[i] - M) ** 2)
    return res


# D(x) = x^2 * f(x) - M^2(x)
def Disp2(x, P):
    res = 0
    for i in range(len(P)):
        res += (x[i] ** 2) * P[i]
    res -= MathWait(P) ** 2
    return res


x = [[], [], []]
y = [[], [], []]
y1 = [[], [], []]


for i in range(len(EXP)):
    l = EXP[i] * p
    # x[i] = [j for j in range((EXP[i] + 1))]     # Мы достанем чёрный шар k раз
    x[i] = [j for j in range(min(EXP[i] + 1, 200))]     # Мы достанем чёрный шар k раз
    # y[i] = []                                   # Вероятность, что мы достанем чёрный шар i раз


    last = 0
    for j in x[i]:
        if j == 0:
            t = exp(-l)
            y[i].append(t)
            last = t
        else:
            t = last * l / j
            y[i].append(t)
            last = t

    print(f"Sum(P[{i}]) = {sum(y[i])}")

    y1[i].append(y[i][0])
    for j in range(1, len(y[i])):
        y1[i].append(y1[i][j - 1] + y[i][j])

    M = MathWait(y[i])

    print(f"M(k) = {M}")

    D1 = Disp1(x[i], y[i])
    D2 = Disp2(x[i], y[i])

    if abs(D1 - D2) > 1e-4:
        print("Bad")
        print(f"D1(k) = {D1}, D2(k) = {D2}")
    else:
        print(f"D(k) = {D1}")


# color = ["r", "g", "b"]

# for i in range(len(EXP)):
#     plt.title("P(k)")  # заголовок
#     plt.xlabel("k")  # ось абсцисс
#     plt.ylabel("P(k)")  # ось ординат
#     plt.grid()  # включение отображение сетки
#     plt.bar(x[i], y[i])  # построение графика
#     plt.show()
#
#     plt.title("F(k)")  # заголовок
#     plt.xlabel("k")  # ось абсцисс
#     plt.ylabel("F(k)")  # ось ординат
#     plt.grid()  # включение отображение сетки
#     plt.step(x[i], y1[i])  # построение графика
#     plt.show()
#
# exit(0)

n = [0.7, 0.8, 0.9, 0.95, 0.99]
f = []
# P = 0.5 - FI(x1) => FI(x1) = 0.5 - P => FI(x1) = integrate(1.0 / sqrt(2 * pi) * exp((-x ** 2) / 2))

# F(n) : P(k >= 3) = 0.7
# qq = [i for i in range(3, 1000)]
# ww = [P(3, i, i) for i in qq]
#
# for i in n:
#     for j in range(len(ww)):
#         if ww[j] > i:
#             f.append(j)
#             print(f"P(3 <= k) >= {i}, n = {j}")
#             break

# plt.title("P(3 <= k)")  # заголовок
# plt.xlabel("n")  # ось абсцисс
# plt.ylabel("P(k)")  # ось ординат
# plt.grid()  # включение отображение сетки
# plt.step(qq, ww)
# plt.show()
#
# plt.title("n(P)")  # заголовок
# plt.xlabel("P(3 <= k)")  # ось абсцисс
# plt.ylabel("n")  # ось ординат
# plt.grid()  # включение отображение сетки
# plt.plot(n, f)
# plt.show()

x = []
y = []
was = [0, 0, 0, 0, 0]
pr = [0, 0, 0, 0, 0]

for i in range(1, 1000):
    lam = i * p
    t = exp(-lam)
    res = 1 - t * (1 + lam + (lam ** 2) / 2)
    x.append(i)
    y.append(res)
    for j in range(len(n)):
        if res >= n[j] and was[j] == 0:
            print(f"P(3 <= k) >= {n[j]}, n = {i}")
            was[j] = n[j]
            pr[j] = i
    if abs(res - 1) < 1e-3:
        break


plt.title("P(3 <= k)")  # заголовок
plt.xlabel("n")  # ось абсцисс
plt.ylabel("P(3 <= k)")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot(x, y)
plt.show()


plt.title("")  # заголовок
plt.xlabel("")  # ось абсцисс
plt.ylabel("")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot(was, pr)
plt.show()
