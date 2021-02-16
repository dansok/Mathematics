import matplotlib.pyplot as plt
import numpy as np


def flip_coin():
    return np.random.randint(0, 2)


def main():
    num_flips = 300
    epsilon = 0.1
    bound = 1 / (4 * (epsilon ** 2))
    mu = 0.5
    sim_runs = 1000
    upper_bounds = [bound / i for i in range(1, num_flips + 1)]
    deviations = np.array([0] * num_flips)

    last_average = 0
    for _ in range(sim_runs):
        for i in range(num_flips):
            x = flip_coin()
            # we may compute the averages vector using dynamic programming
            average = (i * last_average + x) / (i + 1)
            last_average = average
            if np.abs(average - mu) > epsilon:
                deviations[i] += 1.

    plt.xlabel('Number of flips')
    plt.ylabel('Probability of error deviating by a quantity greater than epsilon')
    probabilities = deviations / sim_runs
    xaxis_vals = np.arange(num_flips, step=10)
    yaxis_vals = [probabilities[i] for i in xaxis_vals]
    plt.scatter(xaxis_vals, yaxis_vals, color='b', label='Actual results')
    # theoretical upper bounds
    plt.plot(upper_bounds, color='r', label='Theoretical upper bound')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
