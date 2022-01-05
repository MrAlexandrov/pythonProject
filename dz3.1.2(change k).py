import matplotlib.pyplot as plt

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


# def que(n):
#     x = [i for i in range(1, n + 1)]
#     y = [Q[n][i] for i in x]
#     plt.title(f"L оч, n = {n}")  # заголовок
#     plt.xlabel("k")  # ось абсцисс
#     plt.ylabel("L")  # ось ординат
#     plt.grid()  # включение отображение сетки
#     plt.plot(x, y)
#     plt.show()
#     plt.legend()


def Potk(n):
    x = [i for i in range(k + 1)]
    y = [P0Q[n][i] * Q[n][i] for i in x]
    # y[0] = 1
    plt.title(f"P отк")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"n={n}")
    plt.legend()


def Mzan1(n, k):
    res = 0
    for i in range(n + 1):
        res += i * P0Q[n][k] * A[i]
    for i in range(1, k + 1):
        res += n * P0Q[n][k] * Q[n][i]
    return res


def Mzan(k):
    x = [i for i in range(1, n + 1)]
    y = [Mzan1(i, k) for i in x]
    plt.title(f"M зан")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k = {k}")
    plt.legend()


def Kzagr1(n, k):
    return Mzan1(n, k) / n


def Kzagr(k):
    x = [i for i in range(1, n + 1)]
    y = [Kzagr1(i, k) for i in x]
    plt.title(f"K загр")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("K")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k = {k}")
    plt.legend()


def Pque1(n, k):
    res = 1
    for i in range(n + 1):
        res -= P0Q[n][k] * A[i]
    return res


def Pque(k):
    x = [i for i in range(n + 1)]
    y = [Pque1(i, k) for i in x]
    y[0] = 1
    plt.title(f"P оч")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("P")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k = {k}")
    plt.legend()


def MlenQ1(n, k):
    res = 0
    for j in range(1, k + 1):
        res += j * Q[n][j] * P0Q[n][k]
    return res


def MlenQ(k):
    x = [i for i in range(n + 1)]
    y = [MlenQ1(i, k) for i in x]
    # y[0] = 0
    plt.title(f"M длины очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x[1:], y[1:], label=f"k = {k}")
    plt.legend()


def KzanQ1(n, k):
    return MlenQ1(n, k) / k


def KzanQ(k):
    x = [i for i in range(1, n + 1)]
    y = [KzanQ1(i, k) for i in x]
    # y[0] = 0
    plt.title(f"K занятости очереди")  # заголовок
    plt.xlabel("n")  # ось абсцисс
    plt.ylabel("M")  # ось ординат
    plt.grid()  # включение отображение сетки
    plt.plot(x, y, label=f"k = {k}")
    plt.legend()


for i in range(1, k + 1):
    Potk(i)
plt.show()

for i in range(1, k + 1):
    Mzan(i)
plt.show()

for i in range(1, k + 1):
    Kzagr(i)
plt.show()

for i in range(1, k + 1):
    Pque(i)
plt.show()

for i in range(1, k + 1):
    MlenQ(i)
plt.show()

for i in range(1, k + 1):
    KzanQ(i)
plt.show()