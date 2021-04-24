from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from decorator import contextmanager

NUM_PARTICLES = 5
NUM_REACTIONS = 8

displacements = []
t_max = 10


def flip_coin(probability):
    u = np.random.uniform(low=0, high=1)

    if probability <= u:
        return 0

    return 1


@contextmanager
def catch_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


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
    return np.array([lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7], dtype=np.float64)


def tau_leaping(X_0, k, h):
    X = np.array(X_0)
    # previous_X = X.copy()
    taus = np.zeros(NUM_REACTIONS)

    x_axis = []
    y_axis = [[] for _ in X]
    num_intervals = int(t_max / h)

    # Y = np.zeros(NUM_PARTICLES)
    # cumulative_taus = np.zeros(NUM_REACTIONS)

    for n in range(num_intervals):
        x_axis.append(n)
        for i, x in enumerate(X):
            y_axis[i].append(x)

        # previous_taus = taus.copy()

        # cumulative_taus += get_intensities(X=X, k=k)

        taus += h * get_intensities(X=X, k=k)

        # jump_probabilities = [1 - np.exp(-(tau - previous_tau)) for tau, previous_tau in zip(taus, previous_taus)]

        # jumps = [jump_probability * displacements[i] for i, jump_probability in enumerate(jump_probabilities)]
        jumps = [np.random.poisson(taus[l]) * displacements[l] for l in range(NUM_REACTIONS)]

        total_jump = h * np.sum(jumps, axis=0)

        # Y += total_jumps

        # previous_X = X
        X = X + total_jump
        X = np.where(X < 0, 0, X)

    plt.plot(x_axis, y_axis[0], label='G')
    plt.plot(x_axis, y_axis[1], label='M')
    plt.plot(x_axis, y_axis[2], label='P')
    plt.plot(x_axis, y_axis[3], label='D')
    plt.plot(x_axis, y_axis[4], label='B')
    plt.legend()
    plt.show()


def compute_displacements():
    # species G, M, P, D, B
    displacements.append(np.array([0, 1, 0, 0, 0]))  # G -> G + M
    displacements.append(np.array([0, 0, 1, 0, 0]))  # M -> M + P
    displacements.append(np.array([0, -1, 0, 0, 0]))  # M -> {}
    displacements.append(np.array([0, 0, -1, 0, 0]))  # P -> {}
    displacements.append(np.array([0, 0, -2, 1, 0]))  # 2P -> D
    displacements.append(np.array([0, 0, 0, -1, 0]))  # D -> {}
    displacements.append(np.array([-1, 0, 0, -1, 1]))  # G + D -> B
    displacements.append(np.array([1, 0, 0, 1, -1]))  # B -> G + D


def main():
    compute_displacements()

    with catch_time() as time:
        tau_leaping(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 0, 0], h=0.01)
        # tau_leaping(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 2, 0.1], h=0.01)

        print(f'time: {time():.4f} seconds')


if __name__ == '__main__':
    main()
