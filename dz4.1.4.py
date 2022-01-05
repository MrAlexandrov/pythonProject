import matplotlib.pyplot as plt

R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]

a = 41
c = 1
X0 = B[1]
m = 100
was = [-1 for i in range(m + 1)]
x = [X0]

was[X0] = 0
for i in range(1, 2 * m):
    now = (a * x[-1] + c) % m
    x.append(now)
    if was[now] != -1:
        print(f"Period is equal {i - was[now]}");
        break
    was[now] = i
# for i in range(len(x)):
#     print(f"{i} {x[i]}")

k = 10

gist = [0 for i in range(k)]


def seg(x):
    return x // k


# nn = [i for i in range(2, 101)]
nn = [5, 10, 25, 50]

M = []
S = []
D1 = []
D2 = []

for n in nn:
    print(f"n = {n}")
    for i in range(n):
        gist[seg(x[i])] += 1


    # plt.title(f"")  # заголовок
    # plt.xlabel("r")  # ось абсцисс
    # plt.ylabel("cnt")  # ось ординат
    # plt.grid()  # включение отображение сетки
    # plt.bar([i for i in range(len(gist))], gist)
    # plt.show()


    xi2 = 0
    p = 1 / k
    for i in range(k):
        xi2 += ((gist[i] - n * p) ** 2) / (n * p)

    print(f"xi2 = {xi2}")
    M.append(xi2)
    # alpha = 0.10

    sr = 0
    for i in range(n):
        sr += x[i]
    sr /= n
    S.append(sr)
    print(f"Выборочное среднее равно = {sr}")

    disp = 0
    for i in range(n):
        disp += (x[i] - sr) ** 2
    disp /= n
    D1.append(disp)
    print(f"Выборочная дисперсия равна = {disp}")

    Sn = 0
    for i in range(n):
        Sn += (x[i] - sr) ** 2
    Sn /= n - 1
    D2.append(Sn)
    print(f"Исправленная оценка выборочной дисперсии равна = {Sn}")

plt.title(f"Xi^2")  # заголовок
plt.xlabel("n")  # ось абсцисс
plt.ylabel("")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot([0] + nn, [0] + M)
plt.show()

plt.title(f"Выборочное среднее")  # заголовок
plt.xlabel("n")  # ось абсцисс
plt.ylabel("")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot([0] + nn, [0] + S)
plt.show()

plt.title(f"Смещённая выборочная дисперсия")  # заголовок
plt.xlabel("n")  # ось абсцисс
plt.ylabel("")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot([0] + nn, [0] + D1)
plt.show()

plt.title(f"Исправленная выборочная дисперсия")  # заголовок
plt.xlabel("")  # ось абсцисс
plt.ylabel("")  # ось ординат
plt.grid()  # включение отображение сетки
plt.plot([0] + nn, [0] + D2)
plt.show()