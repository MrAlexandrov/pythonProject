import matplotlib.pyplot as plt

R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]

n = G[1] + B[1] + R[2] + B[2] + R[3] + G[3]
Tc = RGB[1] + RGB[2] + RGB[3]
Ts = R[1] + G[2] + B[3]

l = 1 / Tc      # интенсивность потока заявок
m = 1 / Ts      # интенсивность потока обслуживания
p = l / m

print(f"n = {n}")
print(f"Tc = {Tc}")
print(f"Ts = {Ts}")

A = [1]
P0 = 1

Mstay = []
Mwait = []
Pwait = []
Mbusy = []
Kbusy = []


def FMstay(x):
    res = 0
    for i in range(n + 1):
        res += i * A[i]
    res *= P0
    return res


def FMwait(x):
    res = 0
    for j in range(x + 1, n + 1):
        res += (j - x) * A[j]
    res *= P0
    return res


def FPwait(x):
    res = 0
    for j in range(x + 1, n + 1):
        res += A[j]
    res *= P0
    return res


def FMbusy(x):
    res = 0
    for i in range(x + 1):
        res += i * A[i]
    for j in range(x + 1, n + 1):
        res += x * A[j]
    res *= P0
    return res


def FKbusy(x):
    return FMbusy(x) / x


for M in range(1, n + 1):
    A = [1]
    sm = 1
    for i in range(1, M + 1):
        A.append(A[-1] * p * (n - i + 1) / i)
        sm += A[-1]
    for j in range(M + 1, n + 1):
        A.append(A[-1] * p * (n - j + 1) / M)
        sm += A[-1]
    P0 = 1 / sm
    Mstay.append(FMstay(M))
    Mwait.append(FMwait(M))
    Pwait.append(FPwait(M))
    Mbusy.append(FMbusy(M))
    Kbusy.append(FKbusy(M))

# Матожидание простаивающих и матожидание ожидающих станков
# x = [i for i in range(1, n + 1)]
# plt.title(f"M stay")  # заголовок
# plt.xlabel("m")  # ось абсцисс
# plt.ylabel("M")  # ось ординат
# plt.grid()  # включение отображение сетки
# plt.plot(x, Mstay, label = f"Mstay")
# plt.plot(x, Mwait, label = f"Mwait")
# plt.legend()
# plt.show()


x = [i for i in range(1, n + 1)]
y = Mstay
plt.title(f"M stay")  # заголовок
plt.xlabel("m")  # ось абсцисс
plt.ylabel("M")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar(x, y)
plt.show()

y = Mwait
plt.title(f"M wait")  # заголовок
plt.xlabel("m")  # ось абсцисс
plt.ylabel("M")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar(x, y)
plt.show()

y = Pwait
plt.title(f"P wait")  # заголовок
plt.xlabel("m")  # ось абсцисс
plt.ylabel("P")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar(x, y)
plt.show()

y = Mbusy
plt.title(f"M busy")  # заголовок
plt.xlabel("m")  # ось абсцисс
plt.ylabel("M")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar(x, y)
plt.show()

y = Kbusy
plt.title(f"K busy")  # заголовок
plt.xlabel("m")  # ось абсцисс
plt.ylabel("k")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar(x, y)
plt.show()