import matplotlib.pyplot as plt
import numpy as np


displacements = []
t_max = 10


def get_intensities(X, k):
    # species G, M, P, D, B
    lambda_0 = k[0] * X[0]  # G -> G + M
    lambda_1 = k[1] * X[1]  # M -> M + P
    lambda_2 = k[2] * X[1]  # M -> {}
    lambda_3 = k[3] * X[2]  # P -> {}
    lambda_4 = k[4] * X[2] * (X[2] - 1)  # 2P -> D
    lambda_5 = k[5] * X[3]  # D -> {}
    lambda_6 = k[6] * X[0] * X[3]  # G + D -> B
    lambda_7 = k[7] * X[4]  # B -> G + D
    return [lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7]


def get_l_star(intensities):
    u = np.random.uniform(0, sum(intensities))
    running_total = 0
    for i, intensity in enumerate(intensities):
        running_total += intensity
        if u < running_total:
            return i


def gillespie(X_0, k):
    X = np.array(X_0)
    t = 0

    x_axis = []
    y_axis = [0]
    for value in X:
        x_axis.append([value])

    while t < t_max:
        intensities = get_intensities(X, k)
        r = np.random.exponential(1 / sum(intensities))
        t += r
        l_star = get_l_star(intensities)
        X += displacements[l_star]

        y_axis.append(t)
        for i, value in enumerate(X):
            x_axis[i].append(value)

    print(t, X)
    plt.plot(y_axis, x_axis[0], label='G')
    plt.plot(y_axis, x_axis[1], label='M')
    plt.plot(y_axis, x_axis[2], label='P')
    plt.plot(y_axis, x_axis[3], label='D')
    plt.plot(y_axis, x_axis[4], label='B')
    plt.legend()
    plt.show()


def compute_displacements():
    # species G, M, P, D, B
    displacements.append([0, 1, 0, 0, 0])  # G -> G + M
    displacements.append([0, 0, 1, 0, 0])  # M -> M + P
    displacements.append([0, -1, 0, 0, 0])  # M -> {}
    displacements.append([0, 0, -1, 0, 0])  # P -> {}
    displacements.append([0, 0, -2, 1, 0])  # 2P -> D
    displacements.append([0, 0, 0, -1, 0])  # D -> {}
    displacements.append([-1, 0, 0, -1, 1])  # G + D -> B
    displacements.append([1, 0, 0, 1, -1])  # B -> G + D


def main():
    compute_displacements()

    gillespie(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 0, 0])
    gillespie(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 2, 0.1])


if __name__ == '__main__':
    main()
