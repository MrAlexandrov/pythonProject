import matplotlib.pyplot as plt

R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]

Tc = R[1]
Ts = G[1] + B[1] + RGB[2]
Tw = RGB[1] + RGB[2] + RGB[3]

l = 1 / Tc      #
m = 1 / Ts      # интенсивность потока обслуживания
d = 1 / Tw
p = l / m
A = [1] #       1 / 0!,      p / 1!,              p^2 / 2!,                p^3 / 3!,        p^4 / 4!, ...


for i in range(1, 13):
    A.append(A[-1] * p / i)


def sumAn(n):
    res = 0
    for i in range(n + 1):
        res += A[i]
    return res


n = 12
k = 40


def coef(n, k):
    return l / (n * m + k * d)


W = [[0] * k for i in range(n + 1)]
sumW = [0 for i in range(n + 1)]

for i in range(1, n + 1):
    W[i][0] = 1
    for j in range(1, k):
        W[i][j] = W[i][j - 1] * coef(i, j)
        sumW[i] += W[i][j]


P0INFW = [1]
for i in range(1, n + 1):
    sm = sumAn(i)
    sm += A[i] * sumW[i]
    P0INFW.append(1 / sm)

st = 1


def Mzan1(n):
    res = 0
    for i in range(n + 1):
        res += i * A[i]
    res += n * A[n] * sumW[n]
    res *= P0INFW[n]
    return res


def Mzan(n):
    x = [i for i in range(st, n + 1)]
    y = [Mzan1(i) for i in x]
    plt.title(f"M зан")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.bar(x, y)
    plt.show()


def Kzagr1(n):
    return Mzan1(n) / n


def Kzagr(n):
    x = [i for i in range(st, n + 1)]
    y = [Kzagr1(i) for i in x]
    plt.title(f"K загр")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.bar(x, y)
    plt.show()


def Pque1(n):
    res = 1
    for i in range(n + 1):
        res -= P0INFW[n] * A[i]
    return res


def Pque(n):
    x = [i for i in range(st, n + 1)]
    y = [Pque1(i) for i in x]
    # y1 = [P0INFW[i] * A[i] * (p / i) / (1 - (p / i)) for i in x]
    plt.title(f"P оч")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.bar(x, y)
    # plt.plot(x, y1, label=f"k = {n}")
    plt.show()


def MlenQ1(n):
    t = 0
    for i in range(1, len(W)):
        t += i * W[n][i]
    return A[n] * P0INFW[n] * t


def MlenQ(n):
    x = [i for i in range(st, n + 1)]
    y = [MlenQ1(i) for i in x]
    plt.title(f"M длины очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.bar(x, y)
    plt.show()


Mzan(n)
Kzagr(n)
Pque(n)
MlenQ(n)