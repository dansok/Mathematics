import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

upper = 30
trials_size = 10000
sample_size = 20


def get_sample_uniform():
    return [np.random.uniform(0, upper) for _ in range(sample_size)]


def main():
    results = []
    counts = [0] * (upper + 1)
    for _ in range(trials_size):
        sample = [np.random.uniform(0, upper) for _ in range(sample_size)]
        x = int(np.round(np.average(sample)))
        counts[x] += 1
        results.append(np.average(sample))
    probabilities = [x / trials_size for x in counts]
    xaxisvals = np.arange(upper + 1)
    plt.xlabel('Sampling')
    plt.ylabel('Probability')
    plt.scatter(xaxisvals, probabilities)
    # plt.hist(probablities, 10)

    plot_normal_distribution()
    plt.show()


def plot_normal_distribution():
    # for the uniform distribution 
    # variance = 1/12 * (b - a) ** 2 = 8.333333
    # sigma = sqrt(var)
    # standard deviation = sigma / sqrt(sample_size)

    variance = 1 / 12 * (upper ** 2)
    sigma = np.sqrt(variance)
    sd = sigma / np.sqrt(sample_size)
    mean = upper / 2
    x_axis = np.arange(0, upper, 0.01)
    plt.plot(x_axis, norm.pdf(x_axis, mean, sd), color='r')


if __name__ == '__main__':
    main()
