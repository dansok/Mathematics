import random
import numpy as np
import matplotlib.pyplot as plt


epsilon = 0.1
variance = 0.25  # coin flip is Bernoulli distribution
sigma = np.sqrt(variance)
sg1 = (sigma / epsilon) ** 2

epochsPerNumber = 100 
step = 50

def flip_coin():
    return random.randint(0,1)

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

    probabilities = [averages[-1]]
    upper_bounds = [bound]
    for i, x in enumerate(xs[1:], start=1):
        average = (i * averages[-1] + x) / (i+1)
        averages.append(average)
        probabilities.append(average / (i + 1))
        upper_bounds.append(bound / (i+1))

        if np.abs(averages[-1] - mu) > epsilon:
            successes += 1

    plt.xlabel('Number of flips')
    plt.ylabel('Probability of error being out of range')
    plt.plot(probabilities, color='b', linestyle='dotted')
    plt.plot(upper_bounds, color='r', linestyle='dotted')
    plt.show()

    # print(f'probabilities: {probabilities[:100]}')
    print(f'empirical probability = {successes / num_runs}')


if __name__ == '__main__':
    main()

