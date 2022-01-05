import matplotlib.pyplot as plt
from math import *

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

print(Tc, Ts)
print(p)

sm = A[0]

for i in range(1, 100):
    A.append(A[-1] * p / i)
    sm += A[-1]


def sumAn(n):
    res = 0
    for i in range(n + 1):
        res += A[i]
    return res


n = 12
k = 12

sm = 0

P0INF = [1]
for i in range(1, n + 7):
    sm = sumAn(i)
    a = p / i
    sm += A[i] * a / (1 - a)
    P0INF.append(1 / sm)

def Mzan1(n):
    res = 0
    for i in range(n + 1):
        res += i * P0INF[n] * A[i]
    a = p / n
    res += P0INF[n] * (n * (p ** n) / factorial(n)) * (a / (1 - a))
    return res


def Mzan(n):
    x = [i for i in range(6, n + 1)]
    y = [Mzan1(i) for i in x]
    plt.title(f"M зан")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def Kzagr1(n):
    return Mzan1(n) / n


def Kzagr(n):
    x = [i for i in range(6, n + 1)]
    y = [Kzagr1(i) for i in x]
    plt.title(f"K загр")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


def sumPINF(n):
    res = P0INF[n]
    for i in range(1, n + 1):
        res += A[i] * P0INF[n]
    return res


def Pque1(n):
    res = 1
    for i in range(n + 1):
        res -= P0INF[n] * A[i]
    return res


def Pque(n):
    x = [i for i in range(6, n + 1)]
    y = [Pque1(i) for i in x]
    # y1 = [P0INFW[i] * A[i] * (p / i) / (1 - (p / i)) for i in x]
    plt.title(f"P оч")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    # plt.plot(x, y1, label=f"k = {n}")
    plt.show()


def MlenQ1(n):
    a = p / n
    return A[n] * P0INF[n] * a / ((1 - a) ** 2)


def MlenQ(n):
    x = [i for i in range(6, n + 1)]
    y = [MlenQ1(i) for i in x]
    plt.title(f"M длины очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y)
    plt.show()


Mzan(n)
Kzagr(n)
Pque(n)
MlenQ(n)