import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


histogram_bins = np.arange(0, 5, 0.25)


def get_uniform_sample():
    return np.random.uniform(0, 1)


def main():
    dimensions = 2

    invariant_mean = np.zeros(dimensions)
    invariant_sigma = 1
    invariant_mvn = get_mvn(dimensions, invariant_mean, invariant_sigma)

    proposal_sigma = 1
    proposal_cov = proposal_sigma * np.array(np.identity(dimensions))

    # set initial proposal - all dimensions = 1
    current_proposal = np.empty(dimensions)
    current_proposal.fill(1)

    samples = [current_proposal]
    for _ in range(300000):
        proposal = np.random.multivariate_normal(current_proposal, proposal_cov)
        u = get_uniform_sample()
        alpha = min(1, invariant_mvn.pdf(proposal) / invariant_mvn.pdf(current_proposal))
        if u < alpha:
            samples.append(proposal)
            current_proposal = proposal
    print(len(samples))

    # plot normalized histogram of amount of successful samples by distance from the invariant mean
    distances = [np.linalg.norm(x - invariant_mean) for x in samples]
    plt.hist(distances, bins=histogram_bins, density=True, label='MCMC Sampling')

    # invariant normal distribution by distance
    plot_normal_distribution_by_distance(dimensions, invariant_mean, invariant_sigma)
    plt.xlabel('Distance from mean')
    plt.ylabel('Probability')
    plt.title(f'Probability by distance from mean\nNumber of dimensions: {dimensions}')
    plt.legend()
    plt.show()


def get_mvn(dimensions, mean, sigma):
    cov = (sigma ** 2) * np.array(np.identity(dimensions))
    mvn = multivariate_normal(mean, cov)
    return mvn


def plot_normal_distribution_by_distance(dimensions, mean, sigma):
    cov = (sigma ** 2) * np.array(np.identity(dimensions))
    distances = []
    for _ in range(100000):
        x = np.random.multivariate_normal(mean, cov)
        distance = np.linalg.norm(x - mean)
        distances.append(distance)
    plt.hist(distances, bins=histogram_bins, density=True, histtype='step', label='Normal distribution')


if __name__ == '__main__':
    main()
