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
    p = GOOD / ALL
    q = BAD / ALL
    return C(n, k) * (p ** k) * (q ** (n - k))


def local(n, k):
    p = GOOD / ALL
    q = BAD / ALL
    x = (k - n * p) / sqrt(n * p * q)
    return (1 / sqrt(n * p * q)) * fi(x)


def P(k1, k2, n):
    p = GOOD / ALL
    q = BAD / ALL
    x1 = (k1 - n * p) / sqrt(n * p * q)
    x2 = (k2 - n * p) / sqrt(n * p * q)
    return FI(x2) - FI(x1)


def MathWaight(P):
    res = 0
    for i in range(len(P)):
        res += i * P[i]
    return res


# D(x) = f(x) * (x - M) ^ 2
def Disp1(x, P):
    res = 0
    M = MathWaight(P)
    for i in range(len(P)):
        res += P[i] * ((x[i] - M) ** 2)
    return res


# D(x) = x^2 * f(x) - M^2(x)
def Disp2(x, P):
    res = 0
    for i in range(len(P)):
        res += (x[i] ** 2) * P[i]
    res -= MathWaight(P) ** 2
    return res


for i in EXP:
    x = [j for j in range((i + 1))]  # Мы достанем чёрный шар k раз
    y = []  # Вероятность, что мы достанем чёрный шар i раз

    for j in x:
        good = GOOD
        bad = BAD
        all = ALL
        if i <= 1000:
            p = Bernulli(i, j)
        else:
            p = local(i, j)
        y.append(p)

    print(f"Sum(P) = {sum(y)}")

    y1 = [y[0]]
    for i in range(1, len(y)):
        y1.append(y1[i - 1] + y[i])

    M = MathWaight(y)

    print(f"M(k) = {M}")

    D1 = Disp1(x, y)
    D2 = Disp2(x, y)

    if abs(D1 - D2) > 1e-7:
        print("Bad")
        print(f"D1(k) = {D1}, D2(k) = {D2}")
    else:
        print(f"D(k) = {D1}")


    if i == 100:
        l, r = 0, 10
    elif i == 1000:
        l, r = 0, 30
    else:
        l, r = 80, 175
    xx = x[l:r]
    yy = y[l:r]
    yy1 = y1[l:r]


    plt.title("P(k)")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("P(k)")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(xx, yy)  # построение графика
    plt.show()

    plt.title("F(k)")  # заголовок
    plt.xlabel("k")  # ось абсцисс
    plt.ylabel("F(k)")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(xx, yy1)  # построение графика
    plt.show()


n = [0.7, 0.8, 0.9, 0.95, 0.99]
f = []
# P = 0.5 - FI(x1) => FI(x1) = 0.5 - P => FI(x1) = integrate(1.0 / sqrt(2 * pi) * exp((-x ** 2) / 2))

# F(n) : P(k >= 3) = 0.7
qq = [i for i in range(3, 1000)]
ww = [P(3, i, i) for i in qq]

for i in n:
    for j in range(len(ww)):
        if ww[j] > i:
            f.append(j)
            print(f"P(3 <= k) >= {i}, n = {j}")
            break

plt.title("P(3 <= k)")  # заголовок
plt.xlabel("n")  # ось абсцисс
plt.ylabel("P(k)")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot(qq, ww)
plt.show()

plt.title("n(P)")  # заголовок
plt.xlabel("P(3 <= k)")  # ось абсцисс
plt.ylabel("n")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot(n, f)
plt.show()