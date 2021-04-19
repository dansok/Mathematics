import matplotlib.pyplot as plt
import numpy as np

displacements = []
t_max = 10


def calc_intensities(X, k):
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
