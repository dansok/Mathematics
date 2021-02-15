import numpy as np


# A general exponential Random Variable has the pdf f(x) = λ * exp(-λ * x); with mean 1/λ.
# An exponential Random Variable with mean 1, therefore, must satisfy λ = 1.
# Note: l stands for λ (lambda), but this is a reserved python keyword.


l = 1
low = 0
high = 10
interval_length = high - low


def get_weighted_sample():
    x = np.random.uniform(low, high)
    return x * l * np.exp(-l * x)


def monte_carlo_exponential_variable_mean(num_random_vars, sim_runs=1000):
    averages = [np.average([get_weighted_sample() for _ in range(num_random_vars)]) for _ in range(sim_runs)]

    # a monte carlo simulation over $(0, 1)$ is the simple sum $1/N * \sum_{i=1}^{sim_runs} x * f_X(x)$
    # for $x$ drawn uniformly from $(0, 1)$. If we want to simulate over $(0, interval_length)$,
    # we must multiply by the length of of the interval to account for the factor in
    # the change of variables $x ↦ interval_length * x$
    return np.average(averages) * interval_length


def main():
    print(
        'Question #1: Monte Carlo Simulated mean of exponential RVs with mean 1:',
        monte_carlo_exponential_variable_mean(num_random_vars=1000))


if __name__ == '__main__':
    main()
