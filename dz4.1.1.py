R = [2, 8, 11, 5]
G = [1, 10, 6, 10]
B = [2, 9, 11, 7]
RGB = [5, 27, 28, 22]

a = R[1]
c = G[1]
X0 = B[1]
m = 100
was = [-1 for i in range(m + 1)]
x = [X0]

was[X0] = 0
for i in range(1, 2 * m):
    now = (a * x[-1] + c) % m
    x.append(now)
    if was[now] != -1:
        print(f"Period is equal {i - was[now]}")
        break
    was[now] = i

print(f"Total numbers: {len(x)}")

for i in range(len(x)):
    print(f"{i} {x[i]}")

