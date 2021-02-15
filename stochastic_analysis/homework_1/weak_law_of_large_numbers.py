import numpy as np
import matplotlib.pyplot as plt


def flip_coin():
    return np.random.randint(0, 1)


def main():
    successes = 0
    num_runs = 1000000
    epsilon = 0.1
    bound = 1 / (4 * (epsilon ** 2))
    mu = 0.5

    xs = [flip_coin() for _ in range(num_runs)]
    averages = [xs[0]]
    if np.abs(averages[-1] - mu) > epsilon:
        successes += 1

    probabilities = [averages[0]]
    upper_bounds = [bound]
    for i, x in enumerate(xs[1:], start=1):
        # we may compute the averages vector using dynamic programming
        average = (i * averages[-1] + x) / (i + 1)
        averages.append(average)
        probabilities.append(average / (i + 1))
        upper_bounds.append(bound / (i + 1))

        if np.abs(average - mu) > epsilon:
            successes += 1

    plt.xlabel('Number of flips')
    plt.ylabel('Probability of error deviating by a quantity greater than epsilon')
    plt.plot(probabilities, color='b', linestyle='dotted')
    # theoretical upper bounds
    plt.plot(upper_bounds, color='r', linestyle='dotted')
    plt.show()

    # print(f'probabilities: {probabilities[:100]}')
    print(f'empirical probability = {successes / num_runs}')


if __name__ == '__main__':
    main()
