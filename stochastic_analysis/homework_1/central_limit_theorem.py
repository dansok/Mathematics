import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

upper = 30
trials_size = 10000
sample_size = 20


def get_sample_uniform():
    return [np.random.uniform(0, upper) for _ in range(sample_size)]


def main():
    counts = np.array([0] * (upper + 1))
    for _ in range(trials_size):
        sample = get_sample_uniform()
        x = int(np.round(np.average(sample)))
        counts[x] += 1
    probabilities = counts / trials_size
    xaxis_vals = np.arange(upper + 1)
    plt.xlabel('Sampling')
    plt.ylabel('Probability')
    plt.scatter(xaxis_vals, probabilities, label='Actual results')
    plot_normal_distribution()
    plt.legend()
    plt.show()


def plot_normal_distribution():
    # for a uniform distribution
    # variance = 1/12 * (b - a) ** 2, in our case b = upper, a = 0
    # sigma = sqrt(variance)
    # standard deviation = sigma / sqrt(sample_size)

    variance = 1 / 12 * (upper ** 2)
    sigma = np.sqrt(variance)
    sd = sigma / np.sqrt(sample_size)
    mean = upper / 2
    x_axis = np.arange(0, upper, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, mean, sd), color='r', label='Normal distribution')


if __name__ == '__main__':
    main()
