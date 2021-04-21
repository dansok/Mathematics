import matplotlib.pyplot as plt
import numpy as np


NUM_PARTICLES = 5
NUM_REACTIONS = 8

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


def tau_leaping(X_0, k, h):
    X = np.array(X_0)
    Xs = []
    taus = np.zeros(NUM_REACTIONS)

    x_axis = []
    y_axis = [[] for _ in X]
    num_intervals = int(t_max / h)

    Y = np.zeros(NUM_PARTICLES)

    for n in range(num_intervals):
        x_axis.append(n)
        for i, x in enumerate(X):
            y_axis[i].append(x)

        Xs.append(X)

        previous_taus = taus.copy()

        print(f'Xs: {Xs}')
        for l in range(NUM_REACTIONS):
            taus[l] = 0
            for j in range(n):
                print(f'get_intensities(X=Xs[{j}], k=k): {get_intensities(X=Xs[j], k=k)}')
                taus[l] += (get_intensities(X=Xs[j], k=k)[l])
            taus[l] += (h * get_intensities(X=Xs[n], k=k)[l])
            taus[l] *= h

            jump_probability = 1 - np.exp(-(taus[l] - previous_taus[l]))
            expected_jump = h * jump_probability * displacements[l]

            Y += expected_jump

        X = X + Y

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

    tau_leaping(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 0, 0], h=0.01)
    # tau_leaping(X_0=[1, 10, 50, 10, 0], k=[200, 10, 25, 1, 0.01, 1, 2, 0.1], h=0.5)


if __name__ == '__main__':
    main()
