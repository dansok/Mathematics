from functools import cache

import matplotlib.pyplot as plt
import numpy as np


NUM_REACTIONS = 8


displacements = []
t_max = 10


def G_inverse(l, r, s, X, k):
    intensity = get_intensities(X, k)[l]
    # We take the argmin over all the G^{-1} anyway. Otherwise we get a divide by zero exception.
    if intensity == 0:
        return np.infty

    return s + (r / intensity)


# @cache
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
    return np.array([lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7])


def next_reaction(X_0, S_0, T_1, tau_0, k):
    X = np.array(X_0)
    S = np.array(S_0)
    T = np.array(T_1)
    tau = np.array(tau_0)

    # Since S_0 == [0] * NUM_PARTICLES anyway, l_star could be any index to start out with.
    l_star = 0
    s = S[l_star]

    x_axis = [0]
    y_axis = [[x] for x in X]

    while s < t_max:
        S = [G_inverse(l=l, r=T[l] - tau[l], s=s, X=X, k=k) for l in range(NUM_REACTIONS)]

        l_star = np.argmin(S)
        previous_s = s
        s = S[l_star]

        # Observe that $$\tau_{\ell}^{S_j} = \int_0^{S_j} \lambda_{\ell} (X_s) ds =
        # \sum_{m=0}^{j-1} (S_{m+1} - S_m) \lambda_{\ell}(X_m)$$ since $\{S_i\}$ are the jump times of $X_t$.
        # Hence $\tau$ accumulates in piece-wise sums, as an integral over a step function.
        # We may program this dynamically in the following way -
        # (Note that the expression tau += get_intensities(X=X, k=k) * (s - previous_s) throws a type error here in
        # > python 3.9)
        tau = tau + get_intensities(X=X, k=k) * (s - previous_s)

        X += displacements[l_star]

        T[l_star] -= np.log(np.random.uniform(low=0.0, high=1.0))

        x_axis.append(s)
        for i, x in enumerate(X):
            y_axis[i].append(x)

    print(s, X)
    plt.plot(x_axis, y_axis[0], label='G')
    plt.plot(x_axis, y_axis[1], label='M')
    plt.plot(x_axis, y_axis[2], label='P')
    plt.plot(x_axis, y_axis[3], label='D')
    plt.plot(x_axis, y_axis[4], label='B')
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

    next_reaction(
        X_0=[1, 10, 50, 10, 0],
        S_0=[0] * NUM_REACTIONS,
        T_1=[np.random.exponential() for _ in range(NUM_REACTIONS)],
        tau_0=[0] * NUM_REACTIONS,
        k=[200, 10, 25, 1, 0.01, 1, 0, 0])

    next_reaction(
        X_0=[1, 10, 50, 10, 0],
        S_0=[0] * NUM_REACTIONS,
        T_1=[np.random.exponential() for _ in range(NUM_REACTIONS)],
        tau_0=[0] * NUM_REACTIONS,
        k=[200, 10, 25, 1, 0.01, 1, 2, 0.1])


if __name__ == '__main__':
    main()
