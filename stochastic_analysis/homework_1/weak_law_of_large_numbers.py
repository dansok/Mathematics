import numpy as np
import matplotlib.pyplot as plt


def flip_coin():
    return np.random.randint(0, 1)


def main():
    num_flips = 100
    epsilon = 0.1
    bound = 1 / (4 * (epsilon ** 2))
    mu = 0.5
    sim_runs = 1000
    successes = np.array([[0] * sim_runs] * num_flips)
    upper_bounds = [bound / i for i in range(1, num_flips + 1)]

    last_average = 0
    for j in range(sim_runs):
        for i in range(num_flips):
            x = flip_coin()
            # we may compute the averages vector using dynamic programming
            average = (i * last_average + x) / (i + 1)
            last_average = average
            successes[i][j] = 1. if np.abs(average - mu) > epsilon else 0.

    probabilities = successes.sum(axis=0) / num_flips
    m, n = successes.shape
    print(f'm, n: {m, n}')
    print(f'probabilities: {successes[5000:5005, 50:55]}')
    plt.xlabel('Number of flips')
    plt.ylabel('Probability of error deviating by a quantity greater than epsilon')
    plt.plot(probabilities, color='b', linestyle='dotted')
    # theoretical upper bounds
    plt.plot(upper_bounds, color='r', linestyle='dotted')
    # plt.show()


if __name__ == '__main__':
    main()
