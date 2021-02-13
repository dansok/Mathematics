import numpy as np


class ExponentialRV:

    # A general exponential Random Variable has the pdf f(x) = λ * exp(-λ * x); with mean 1/λ.
    # An exponential Random Variable with mean 1, therefore, must satisfy λ = 1.
    # Note: l stands for λ (lambda), but this is a reserved python keyword.

    def __init__(self, l=1):
        self.pdf = lambda x: l * np.exp(-l * x)

    def get_weighted_sample(self):
        x = np.random.uniform(low=0, high=10)

        return x * self.pdf(x)


def monte_carlo_exponential_variable_mean(num_random_vars, sim_runs=1000):
    random_vars = [ExponentialRV() for _ in range(num_random_vars)]

    return np.average([[random_var.get_weighted_sample() for _ in range(sim_runs)] for random_var in random_vars])


def main():
    print(
        'Question #1: Monte Carlo Simulated mean of exponential RVs with mean 1:',
        monte_carlo_exponential_variable_mean(num_random_vars=100))


if __name__ == '__main__':
    main()
