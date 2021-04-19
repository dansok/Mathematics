import matplotlib.pyplot as plt
import numpy as np

displacements = []
t_max = 10


def calc_intensities(X, k):
    # species G, M, P, D, B
    lmbd0 = k[0] * X[0]  # G -> G + M
    lmbd1 = k[1] * X[1]  # M -> M + P
    lmbd2 = k[2] * X[1]  # M -> {}
    lmbd3 = k[3] * X[2]  # P -> {}
    lmbd4 = k[4] * X[2] * (X[2] - 1)  # 2P -> D
    lmbd5 = k[5] * X[3]  # D -> {}
    lmbd6 = k[6] * X[0] * X[3]  # G + D -> B
    lmbd7 = k[7] * X[4]  # B -> G + D
    return [lmbd0, lmbd1, lmbd2, lmbd3, lmbd4, lmbd5, lmbd6, lmbd7]


def calc_l_star(intensities):
    total_intensities = sum(intensities)
    uniform_sample = np.random.uniform(0, total_intensities)
    running_total = 0
    for i, intensity in enumerate(intensities):
        running_total += intensity
        if uniform_sample < running_total:
            return i


def gillespie(initial_x, k):
    X = np.array(initial_x)
    disp = np.array(displacements)
    t = 0
    x_vals = []
    x_ticks = [0]
    for value in X:
        x_vals.append([value])
    while t < t_max:
        intensities = calc_intensities(X, k)
        total_intensities = sum(intensities)
        r = np.random.exponential(1 / total_intensities)
        t += r
        l_star = calc_l_star(intensities)
        X += disp[l_star]
        x_ticks.append(t)
        for index, value in enumerate(X):
            x_vals[index].append(value)
    print(t, X)
    plt.plot(x_ticks, x_vals[0], label='G')
    plt.plot(x_ticks, x_vals[1], label='M')
    plt.plot(x_ticks, x_vals[2], label='P')
    plt.plot(x_ticks, x_vals[3], label='D')
    plt.plot(x_ticks, x_vals[4], label='B')
    plt.legend()
    plt.show()


def set_displacements():
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
    set_displacements()
    gillespie([1, 10, 50, 10, 0], [200, 10, 25, 1, 0.01, 1, 0, 0])
    gillespie([1, 10, 50, 10, 0], [200, 10, 25, 1, 0.01, 1, 2, 0.1])


if __name__ == '__main__':
    main()
