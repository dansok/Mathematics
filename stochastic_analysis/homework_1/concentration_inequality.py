import matplotlib.pyplot as plt
import numpy as np


def flip_coin():
    return np.random.randint(0, 2)

def get_log_probs(epsilon):
    num_flips = 300
    mu = 0.5
    sim_runs = 1000
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
    probabilities = deviations / sim_runs
    log_probs = [1 / i * np.log(probabilities[i]) for i in range(1, num_flips)]
    return log_probs

def main():
    plt.xlabel('Number of flips')
    plt.ylabel('Log probabilities scaled by 1 / number of flips')
    log_probs = get_log_probs(0.1)
    plt.plot(log_probs, label='Epsilon = 0.1')
    log_probs1 = get_log_probs(0.05)
    plt.plot(log_probs1, label='Epsilon = 0.05')
    log_probs2 = get_log_probs(0.01)
    plt.plot(log_probs2, label='Epsilon = 0.01')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
