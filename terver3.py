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
RED = R[3]
ALL = RGB[3]
NERED = ALL - RED
EXP = ALL


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


# Зафиксируем последний красный шар,
# тогда слева от него могут быть шары любых цветов
# а справа только не красные.
# Мы можем составить функцию от количества шаров слева (+1) F(k),
# в свою очередь, справа будет ALL - k шаров
# Также стоит заметить, что R <= k <= ALL = R + G + B


x = [i for i in range(ALL + 1)]    # Для каждого k
y = []                                  # P(k) - вероятность того, что мы достанем k шаров и закончим


for i in range(len(x)):
    if i < RED:
        y.append(0)
        continue
    red = RED
    nered = NERED
    all = ALL
    exp = i
    p = C(x[i] - 1, red - 1)                           # Нам не важен порядок среди красных, не красных и в точках [...........R]
    # p = 1
    for j in range(RED):                   # Набираем красные шары
        p *= red / all
        red -= 1
        all -= 1
    for j in range(x[i] - RED):                    # Набираем не красные шары
        p *= nered / all
        nered -= 1
        all -= 1
    y.append(p)


print(f"Sum(P) = {sum(y)}")

plt.title("P(k)")                         # заголовок
plt.xlabel("k")                             # ось абсцисс
plt.ylabel("P(k)")                          # ось ординат
plt.grid()                                  # включение отображение сетки
plt.plot(x, y)                            # построение графика
plt.show()

y1 = [y[0]]
for i in range(1, len(y)):
    y1.append(y1[i - 1] + y[i])

plt.title("F(k)")                         # заголовок
plt.xlabel("k")                             # ось абсцисс
plt.ylabel("F(k)")                          # ось ординат
plt.grid()                                  # включение отображение сетки
plt.plot(x, y1)                            # построение графика
plt.show()


def MathWaight(P):
    res = 0
    for i in range(len(P)):
        res += i * P[i]
    return res


M = MathWaight(y)


print(f"M(k) = {M}")


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


D1 = Disp1(x, y)
D2 = Disp2(x, y)


if abs(D1 - D2) > 1e-7:
    print("Bad")
    print(f"D1(k) = {D1}, D2(k) = {D2}")
else:
    print(f"D(k) = {D1}")
