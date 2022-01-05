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
for i in range(len(x)):
    print(f"{i} {x[i]}")

k = 10

gist = [0 for i in range(k)]


def seg(x):
    return x // k


n = 50


for i in range(n):
    gist[seg(x[i])] += 1


plt.title(f"")  # заголовок
plt.xlabel("r")  # ось абсцисс
plt.ylabel("cnt")  # ось ординат
plt.grid()  # включение отображение сетки
plt.bar([i for i in range(len(gist))], gist)
plt.show()


xi2 = 0
p = 1 / k
for i in range(k):
    xi2 += ((gist[i] - n * p) ** 2) / (n * p)

print(f"xi2 = {xi2}")

# alpha = 0.20