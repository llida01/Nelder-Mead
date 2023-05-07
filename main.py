import numpy as np
from random import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json


class Function:
    def __init__(self, f, n):
        self.n = n
        self.f = f

    def __call__(self, point):
        for i in range(self.n):
            locals()[f'x{i+1}'] = point[i]
        f = eval(self.f)
        return f


fig = plt.figure()
axes = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
ln, = axes.plot([], [], lw=0.8)
x1, y1 = [], []


def init():
    ln.set_data([], [])
    return ln,


def update(i):
    plt.title(f"Step: {i + 1}", )
    if i > len(x1) - 1:
        i = len(x1) - 1
    ln.set_data(x1[i], y1[i])
    return ln,


def Points(n, simplex):
    points = []
    for i in range(n + 1):
        points.append(np.array(simplex[i]))
    return points


def CreatePoints(n):
    points = []
    for i in range(n + 1):
        coordinates = []
        for j in range(n):
            coordinates.append(randint(0, 2))
        point = np.array(coordinates)
        points.append(point)
    return points


def Distance(point1, point2):
    res = 0
    for i in range(len(point1)):
        res = res + (point1[i] - point2[i]) ** 2
    return sqrt(res)


def Squeeze(f, x_w, x_c, x_b, points, betta):
    x_s = betta * x_w + (1 - betta) * x_c
    if f(x_s) < f(x_w):
        points[-1] = x_s
        return points

    elif f(x_s) > f(x_w):
        for j in range(1, len(points) - 1):
            points[j] = x_b + (points[j] - x_b) / 2
        return points


def Neldermead(n, f, alpha, betta, gama, eps, steps, points):
    file = open('result1.txt', 'w')
    last = np.zeros(n)
    count_last = 0
    for i in range(steps):
        points = sorted(points, key=lambda x: f(x))
        x_b = points[0]
        x_w = points[-1]
        x_g = points[-2]
        x_c = np.zeros(n)
        for j in range(len(points) - 1):
            x_c = x_c + points[j]
        x_c = x_c / n
        x_r = (1 + alpha) * x_c - alpha * x_w
        if f(x_r) < f(x_b):
            x_e = (1 - gama) * x_c + gama * x_r

            if f(x_e) < f(x_r):
                x_w = x_e
                points[-1] = x_e

            elif f(x_r) < f(x_e):
                x_w = x_r
                points[-1] = x_r

        elif f(x_b) < f(x_r) < f(x_g):
            points[-1] = x_r

        elif f(x_g) < f(x_r) < f(x_w):
            x_w = x_r
            points[-1], x_r = x_r, points[-1]
            points = Squeeze(f, x_w, x_c, x_b, points, betta)
            x_b = points[0]
            x_w = points[-1]
            x_g = points[-2]

        elif f(x_w) < f(x_r):
            points = Squeeze(f, x_w, x_c, x_b, points, betta)
            x_b = points[0]
            x_w = points[-1]
            x_g = points[-2]
        file.write(f'Step {i + 1}: x_b = {points[0]}, f(x_b) = {f(points[0])} \n')
        x1.append([x_b[0], x_w[0], x_g[0], x_b[0]])
        y1.append([x_b[1], x_w[1], x_g[1], x_b[1]])
        if (Distance(x_b, x_w) <= eps) | (Distance(x_b, x_g) <= eps) | (Distance(x_g, x_w) <= eps):
            break
        if np.array_equal(x_b, last):
            count_last += 1
            if count_last == 10:
                break
        else:
            last = x_b
            count_last = 0
    file.write(f'X_min = {np.around(points[0], 2)}, f(X_min) = {f(np.around(points[0], 2))}')
    file.close()
    return np.around(points[0], 2)


def main():
    with open('settings1.json') as j:
        file = json.load(j)
    n = file['n']
    f = file['f']
    alpha = file['alpha']
    betta = file['betta']
    gama = file['gama']
    eps = file['eps']
    steps = file['steps']
    answer = file['answer']
    if file['start_simplex']:
        simplex = file['simplex']
        start_simplex = Points(n, simplex)
    else:
        start_simplex = CreatePoints(n)
    plt.suptitle(f, fontsize=15)
    f = Function(f, n)
    Neldermead(n, f, alpha, betta, gama, eps, steps, start_simplex)
    plt.plot(answer[0], answer[1], 'o', color='maroon')
    animation = FuncAnimation(fig, update, init_func=init, frames=len(x1), interval=200, blit=False)
    animation.save('GIF1.gif', writer='pillow')
    plt.show()


if __name__ == '__main__':
    main()
